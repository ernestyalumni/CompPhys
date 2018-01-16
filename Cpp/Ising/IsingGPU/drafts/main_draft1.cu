/**
 * @file   : main_draft1.cu
 * @brief  : main file draft for 2-dim. Ising in CUDA C++11/14, 
 * @details : 
 * 
 * @author : Ernest Yeung <ernestyalumni@gmail.com>
 * @date   : 20180108    
 * @ref    : M. Hjorth-Jensen, Computational Physics, University of Oslo (2015)
 * https://github.com/CompPhysics/ComputationalPhysics/blob/master/doc/Programs/LecturePrograms/programs/StatPhys/cpp/ising_2dim.cpp
 * https://www.paypal.com/cgi-bin/webscr?cmd=_donations&business=ernestsaveschristmas%2bpaypal%40gmail%2ecom&lc=US&item_name=ernestyalumni&currency_code=USD&bn=PP%2dDonationsBF%3abtn_donateCC_LG%2egif%3aNonHosted 
 * 
 * which won't go through a 3rd. party such as indiegogo, kickstarter, patreon.  
 * Otherwise, I receive emails and messages on how all my (free) material on 
 * physics, math, and engineering have helped students with their studies, 
 * and I know what it's like to not have money as a student, but love physics 
 * (or math, sciences, etc.), so I am committed to keeping all my material 
 * open-source and free, whether or not 
 * sufficiently crowdfunded, under the open-source MIT license: 
 * 	feel free to copy, edit, paste, make your own versions, share, use as you wish.  
 *  Just don't be an asshole and not give credit where credit is due.  
 * Peace out, never give up! -EY
 * 
 * */
/* 
 * COMPILATION TIP
 * nvcc main_draft1.cu ../grid2d/grid2d.cu ../grid2d/sysparam.cu ../dynam/XORMRGgens.cu ../common/gridsetup.cu -o main
 * 
 * */
#include "../grid2d/grid2d.h"  // Spins2d (struct)  
#include "../grid2d/sysparam.h"  // Sysparam, Avg, TransProb, Sysparam_ptr, Avg_ptr, TransProb_ptr, constTransProb
#include "../dynam/XORMRGgens.h" // devStatesXOR, devStatesMRG, devStatesPhilox4_32_10_t
#include "../common/gridsetup.h" // get_maxGridSize()
//#include "../boundary/boundary.h" // periodic boundary conditions as inline __device__ function periodic

#include <algorithm> // std::min

#include <cooperative_groups.h>  // this should go with metropolis.h, initialize_allup_kernel
#include <iostream>
#include <chrono>  


namespace cg = cooperative_groups;  // this should go with metropolis.h, initialize_allup_kernel


/** @fn init_allup_partialsumM
 * @brief initialize spins all up and calculate partial sums for magnetization M
 * @details 1st part of initialize_allup_kernel, 2nd. part is block_sumM
 * */
__device__ int init_allup_partialsumM(int* Sptr,size_t Lx,size_t Ly) {
	int sum=0; // partial sum of the magnetization M

	// global thread index, k_x = 0,1,...N_x*M_x, k_y = 0,1,...N_y*M_y
	unsigned int k_x = threadIdx.x + blockDim.x * blockIdx.x ; 
	unsigned int k_y = threadIdx.y + blockDim.y * blockIdx.y ; 
	unsigned int k = k_x + gridDim.x * blockDim.x * k_y;  

	for (unsigned int idx = k; idx < Lx*Ly/4; idx+= blockDim.x * gridDim.x * blockDim.y * gridDim.y ) {
		reinterpret_cast<int4*>(Sptr)[idx] = {1,1,1,1} 	;
		int4 s4 = ((int4*) Sptr)[idx]; 
		sum += s4.x + s4.y + s4.z + s4.w; 
	}
	
	// process remaining elements
	for (unsigned int idx = k + Lx*Ly/4 *4; idx < Lx*Ly; idx += 4) { 
		Sptr[idx] = 1; 
		sum += Sptr[idx]; 
	}
	return sum;
};

/** @fn blocksumM
 * @brief reduce sum on thread block of partial sums of spins for magnetization M 
 * @details 2nd. part of initialize_allup_kernel, 1st. part is init_allup_partialsumM 
 * */
__device__ int block_sumM(cg::thread_group tg, int* temp, int sumresult) {
	unsigned int lane = tg.thread_rank();
	
	// Each iteration halves number of active threads
	// Each thread adds to partial sum[i] its sum[lane+i] 
	for (unsigned int idx = tg.size()/2; idx >0; idx/=2) 
	{
		// load the array values with this thread block into temp 
		temp[lane] = sumresult; 
		tg.sync(); // wait for all threads to store into temp 
		if (lane<idx) {
			sumresult += temp[lane+idx]; 
		}
		tg.sync(); // wait for all threads to load 
	}
	return sumresult; // note: only thread 0 will return full sum	
};

/** @fn calcE
 * @brief computes E, a summation of all unique nearest neighbor pairs of spins 
 * @details do summation in shared memory, that include halo cells of width 1 "to the right"
 * */
__device__ int calcE(cg::thread_group & tg, int* Sptr, int* temp, size_t Lx, size_t Ly, const float J) {
	int resultE =0;

	const int RAD = 1; // "radius" of "halo" cells, of width 1 (in this case)  

	// old way of thread, block indexing
	unsigned int k_x = threadIdx.x + blockDim.x * blockIdx.x ; 
	unsigned int k_y = threadIdx.y + blockDim.y * blockIdx.y ;

	unsigned int S_x = static_cast<int>(blockDim.x + RAD); 
	unsigned int S_y = static_cast<int>(blockDim.y + RAD); 
	
	unsigned int s_x = threadIdx.x + RAD; // s_x = 1,2,...S_x-1
	unsigned int s_y = threadIdx.y + RAD; // s_y = 1,2,...S_y-1


	// unsigned int k = k_x + gridDim.x * blockDim.x * k_y; 
	
	// what's the point of making modular via cg::thread_group  
//	dim3 tgtidx = tg.thread_index(); // error class "cooperative_groups::__v1::thread_group" has no member "thread_index"

	// use these loops to account for elements not "covered" by the threads in grid that's launched  
	for (unsigned int l_y=k_y,idxy=0; l_y < Ly; idxy++, l_y += blockDim.y *gridDim.y) { 
		for (unsigned int l_x=k_x, idxx=0; l_x < Lx; idxx++, l_x += blockDim.x*gridDim.x ) {
			
			int lx =0; 	// lx gives back global index on lattice grid of spins 
			int ly =0;	// ly gives back global index on lattice grid of spins
	
			/* 0, M_x
			 * 1 
			 * ... 
			 * M_x-1
			 * */
			for (int i = threadIdx.x; i<S_x; i+=static_cast<int>(blockDim.x) ) {
				for (int j = threadIdx.y; j <S_y; j+= static_cast<int>(blockDim.y) ) {
					lx = i + static_cast<int>(blockDim.x*blockIdx.x); 
					ly = j + static_cast<int>(blockDim.y*blockIdx.y); 
			
					/* lx+idxx*gridDim.x*blockDim.x, idxx=0,1,.. to how many multiples of gridDim.x*blockDim.x for 
 					 * multiples of thread grids to "cover" our lattice grid of spins.  
 					 * (lx+idxx*gridDim.x*blockDim.x)%Lx because we want periodic boundary conditions  
 					 * I try to future proof this by using inline function periodic
					 * */
			
					temp[i+j*S_x] = 
						static_cast<float>( 
							Sptr[ periodic((lx+idxx*gridDim.x*blockDim.x),Lx) + 
								blockDim.x * gridDim.x * periodic((ly + idxy*gridDim.y*blockDim.y),Ly) ] );
				}
			}  
	
			if ( l_x >= Lx || l_y >= Ly) {
				return resultE; 
			}
	
			tg.sync(); 
	
			// do the nearest neighbor (unique) pair of spins summation entirely in shared memory
	
			int stencilindex_x = 0; // stencil index in x-direction
			int stencilindex_y = 0; // stencil index in y-direction 
	
			stencilindex_x = s_x - RAD; // = 0,1,...S_x-2 = (M_x+1)-2 = M_x -1
			stencilindex_y = s_y - RAD; 

			// actual calculation of E
			resultE += (-1.f * J) * temp[ stencilindex_x + stencilindex_y * S_x] * 
				(temp[ stencilindex_x + 1 + stencilindex_y * S_x]
					+ temp[ stencilindex_x + (stencilindex_y + 1)*S_x] ); 
		}
	} // END of loops to make threads do "double duty" to cover other elements in our spin lattice grid that wasn't "covered" by our thread grid
	return resultE; 
	
}

__global__ void initialize_allup_kernel(int* Sptr, Sysparam* sysparams, size_t Lx, size_t Ly, const float J) {
	// global thread index, k_x = 0,1,...N_x*M_x, k_y = 0,1,...N_y*M_y
/*	unsigned int k_x = threadIdx.x + blockDim.x * blockIdx.x ; 
	unsigned int k_y = threadIdx.y + blockDim.y * blockIdx.y ; 
	unsigned int k = k_x + gridDim.x * blockDim.x * k_y;  

	for (unsigned int idx = k; idx < Lx*Ly/4; idx+= blockDim.x * gridDim.x * blockDim.y * gridDim.y ) {
		reinterpret_cast<int4*>(Sptr)[idx] = {1,1,1,1} 	;
	}
	
	// process remaining elements
	for (unsigned int idx = k + Lx*Ly/4 *4; idx < Lx*Ly; idx += 4) { 
		Sptr[idx] = 1; }
*/
	// partial sum of spins for magnetization M
	int sum4M = init_allup_partialsumM( Sptr, Lx,Ly); 
	extern __shared__ int temp[]; 
	auto ttb = cg::this_thread_block(); 
	int block_sum = block_sumM(ttb, temp, sum4M) ;

	if (ttb.thread_rank() == 0) {
		atomicAdd(&(sysparams->M), ((float) block_sum)); 
	}

	int threadsumE = calcE(ttb, Sptr, temp, Lx,Ly,J); // for this thread, here's its partial sum contribution to total energy E
	atomicAdd(&(sysparams->E), ((float) threadsumE) ); 

};

/**
 * @fn initialize_allup
 * @brief "driver" function to initialize energy, spin matrix, and magnetization 
 * */
void initialize_allup(Spins2d& spins2d, Sysparam_ptr& sysParams,const std::array<int,3> MAXGRIDSIZES,const dim3 M_is={32,32}) 
{ 
//	size_t L = spins2d.L; // total number of spins of system
	size_t Lx = spins2d.L_is[0]; // total number of spins of system
	size_t Ly = spins2d.L_is[1]; // total number of spins of system
	const float J = spins2d.J; 

	unsigned int RAD = 1; // "radius" or width of "halo" cells needed 

	/* ========== (thread) grid,block dims ========== */ 
	unsigned long MAX_BLOCKS_y = (MAXGRIDSIZES[1] + M_is.y - 1)/ M_is.y; 
	// notice how we're only launching 1/4 of Ly threads in y-direction needed
	unsigned int N_y = std::min( MAX_BLOCKS_y, ((Ly/4 + M_is.y - 1)/ M_is.y)); 
	unsigned long MAX_BLOCKS_x = (MAXGRIDSIZES[0] + M_is.x - 1)/ M_is.x; 
	// notice how we're only launching 1/4 of Lx threads in x-direction needed
	unsigned int N_x = std::min( MAX_BLOCKS_x, ((Lx/4 + M_is.x - 1)/ M_is.x)); 
	dim3 N_is { N_x,N_y }; // single (thread) block dims., i.e. number of threads in a single (thread) block
	int sharedBytes = (M_is.x+RAD)*(M_is.y + RAD)* sizeof(int);
	
	/* ========== END of (thread) grid,block dims ========== */ 

	initialize_allup_kernel<<<N_is,M_is, sharedBytes>>>(spins2d.S.get(),sysParams.d_sysparams.get(),Lx,Ly,J);


}; // end of function initialize_allup



int main(int argc, char* argv[]) {

	constexpr const float initial_temp = 1.f;  // typically 1.
	constexpr const float final_temp = 3.f;  // typically 3.
	constexpr const float tempstep = 0.05f;  // typically 0.05

	// number of spins, related to 2-dim. grid size Lx x Ly
	std::array<size_t, 2> L_is { 1<<9, 1<<9 }; // 1<<10 = 1024 
	std::array<size_t, 2> L_is_small { 4, 4 }; 
	
	Spins2d spins = {L_is};  
	Spins2d spins_small = {L_is_small};  

	std::cout << " L : " << spins.L_is[0]*spins.L_is[1] << std::endl; 

	Sysparam_ptr sysparams_ptr = { initial_temp } ;
	TransProb_ptr transprob_ptr = { initial_temp , 1.f } ;
	Avg_ptr avgs_ptr;  

	Sysparam_ptr sysparams_ptr_small = { initial_temp } ;
	Avg_ptr avgs_ptr_small;  


	/* ***** (thread) grid,block dims ***** */ 
	/* min of N_x, number of (thread) blocks on grid in x-direction, and MAX_BLOCKS allowed is 
	 * determined here */
	size_t MAXGRIDSIZE = get_maxGridSize();  
	auto MAXGRIDSIZES = get_maxGridSizes();
	std::cout << " MAXGRIDSIZE : " << MAXGRIDSIZE << std::endl; 
	
	// (thread) block dims., remember max. no. threads per block is 1024, as of compute capability 5.2
	dim3 M_is { 1<<5, 1<<5 }; 
	size_t L = spins.L_is[0]*spins.L_is[1]; // doesn't output correct values for n = 1<<30    
	unsigned int MAX_BLOCKS_y = (MAXGRIDSIZES[1] + M_is.y - 1)/ M_is.y; 
	// notice how we're only launching 1/4 of L threads
//	unsigned int N_x = min( MAX_BLOCKS, ((L/4 + M_x - 1)/ M_x)); 
//	int sharedBytes = M_x * sizeof(int);  
	/* ***** END of (thread) grid,block dims ***** */ 

	initialize_allup(spins,sysparams_ptr, MAXGRIDSIZES, M_is);

	/* sanity check */
	Sysparam h_sysparams_out ;  
	cudaMemcpy(&h_sysparams_out, sysparams_ptr.d_sysparams.get(), 1*sizeof(Sysparam), cudaMemcpyDeviceToHost);  // possible error have to be of same type
	std::cout << " h_sysparams_out : " << h_sysparams_out.E << " " << h_sysparams_out.M << " " << h_sysparams_out.T << std::endl; 



}
