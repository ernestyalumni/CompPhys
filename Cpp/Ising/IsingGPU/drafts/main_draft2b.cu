/**
 * @file   : main_draft2.cu
 * @brief  : main file draft for 2-dim. Ising in CUDA C++11/14, 
 * @details : 
 * 
 * @author : Ernest Yeung <ernestyalumni@gmail.com>
 * @date   : 20180114    
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
 * nvcc main_draft1.cu ../grid2d/grid2d.cu ../grid2d/sysparam.cu ../dynam/XORMRGgens.cu ../dynam/metropolis.cu ../common/gridsetup.cu -o main
 * 
 * */
#include "../grid2d/grid2d.h"  // Spins2d (struct)  
#include "../grid2d/sysparam.h"  // Sysparam, Avg, TransProb, Sysparam_ptr, Avg_ptr, TransProb_ptr, constTransProb
#include "../dynam/metropolis.h" // initialize_allup
#include "../common/gridsetup.h" // get_maxGridSize()


#include <iostream>
#include <chrono>  

__device__ int calcintDeltaE(int* temp, const unsigned int S_x, const unsigned int S_y, 
	const unsigned int s_x, const unsigned int s_y, const int RAD) 
{ 
//	int resultDeltaE =0; 
//	int stencilindex_x = 0; // stencil index in x-direction
//	int stencilindex_y = 0; // stencil index in y-direction 
	
//	stencilindex_x = s_x ; // = 1,...S_x-1 = (M_x+1)-1 = M_x 
//	stencilindex_y = s_y ; 

	// actual calculation of Delta E
/*	resultDeltaE = 2 * temp[ stencilindex_x + stencilindex_y * S_x] * 
				(temp[ stencilindex_x + 1 + stencilindex_y * S_x]
					+ temp[ stencilindex_x + (stencilindex_y + 1)*S_x] 
					+ temp[ stencilindex_x - 1 + stencilindex_y * S_x]
					+ temp[ stencilindex_x + (stencilindex_y - 1)*S_x] ); 
					* */
	int resultDeltaE = 2 * temp[ s_x + s_y * S_x] * 
				(temp[ s_x + 1 + s_y * S_x]
					+ temp[ s_x + (s_y + 1)*S_x] 
					+ temp[ s_x - 1 + s_y * S_x]
					+ temp[ s_x + (s_y - 1)*S_x] ); 
					
	return resultDeltaE;  
};

__device__ Sysparam spinflips(cg::thread_group & tg, int* Sptr, float * transprob, 
								int* temp, size_t Lx, size_t Ly, const float J, curandState *state) 
{
	Sysparam results_sysparams { 0.f, 0.f, 0.f }; 
	
	const int RAD = 1; // "radius" of "halo" cells, of width 1 (in this case)  

	// old way of thread, block indexing
	unsigned int k_x = threadIdx.x + blockDim.x * blockIdx.x ; 
	unsigned int k_y = threadIdx.y + blockDim.y * blockIdx.y ;

	unsigned int S_x = static_cast<int>(blockDim.x + 2*RAD); 
	unsigned int S_y = static_cast<int>(blockDim.y + 2*RAD); 
	
	unsigned int s_x = threadIdx.x + RAD; // s_x = 1,2,...S_x-2
	unsigned int s_y = threadIdx.y + RAD; // s_y = 1,2,...S_y-2


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
							Sptr[  (lx+idxx*gridDim.x*blockDim.x) % Lx + 
								blockDim.x * gridDim.x * ( (ly + idxy*gridDim.y*blockDim.y) % Ly ) ] );
				}
			}  
	
			if ( l_x >= Lx || l_y >= Ly) {
				return results_sysparams; 
			}
	
			tg.sync(); 

			// global index k
			size_t k = l_x + gridDim.x*blockDim.x * l_y;  

			/* Copy state to local memory for efficiency */
			curandState localState = state[k]; 

			// so-called "checkerboard" - a "checkerboard" pattern is necessitated because the 
			// change in energy Delta E is dependent upon nearest neighbors, the stencil operation, 
			// the energy at present time t.  This is unlike say the finite difference method approximating 
			// partial differentiation equations where we can say the new update is dependent upon values at previous time steps   

	
			// if tg.thread_rank() even
			if ( ( tg.thread_rank() % 2) == 0) 
			{
			// effectively, pick a random spin 
			// use uniform distribution 
				if ( curand_uniform(&localState) < (1.f/ ((float) Lx*Ly)) ) 
				{
					// do the nearest neighbor (unique) pair of spins summation entirely in shared memory
					int intdeltaE = calcintDeltaE(temp, S_x,S_y,s_x,s_y,RAD);					
				
					// roll dice, see if we transition or not, given transprob
					if ( curand_uniform(&localState) <= transprob[intdeltaE +8] ) 
					{
						// instead of loading entire thread block + halo cells to shared memory again, just make single change  

						// Accept!  
						Sptr[ k] = temp[ s_x + s_y * S_x] *= -1; // flip 1 spin and accept new spin config 
//						atomicAdd(&(sysparams->E), ((float) intdeltaE) * J ); 
//						atomicAdd(&(sysparams->M), 2.f*((float) temp[s_x+s_y*S_x]) ); 
						results_sysparams.E += ((float) intdeltaE) * J ; 
						results_sysparams.M += 2.f*((float) temp[s_x+s_y*S_x]) ; 
					}	 
				}
			}
			tg.sync(); 
			// if tg.thread_rank() odd
			if ( (tg.thread_rank() % 2) == 1) 
			{
				if (curand_uniform(&localState) < (1.f/ ((float) Lx*Ly)) ) 
				{
				// do the nearest neighbor (unique) pair of spins summation entirely in shared memory
					int intdeltaE = calcintDeltaE(temp, S_x,S_y,s_x,s_y,RAD);					
				
					// roll dice, see if we transition or not, given transprob
					if ( curand_uniform(&localState) <= transprob[intdeltaE +8] ) 
					{
						// Accept!  
						Sptr[ k] = temp[ s_x + s_y * S_x] *= -1; // flip 1 spin and accept new spin config 
						results_sysparams.E += ((float) intdeltaE) * J ; 
						results_sysparams.M += 2.f*((float) temp[s_x+s_y*S_x]) ; 
					}	 
				}
			}

		}
	} // END of loops to make threads do "double duty" to cover other elements in our spin lattice grid that wasn't "covered" by our thread grid
	return results_sysparams ;  
};


__global__ void metropolis_kernel(int* Sptr, Sysparam* sysparams, float* transprob, size_t Lx, size_t Ly, const float J,
									curandState *state) 
{
	extern __shared__ int temp[]; 
	auto ttb = cg::this_thread_block(); 
	
	// old way
//	unsigned int j = blockIdx.x + gridDim.x * blockIdx.y; // block j out of N_x*N_y total number of (thread) blocks
	dim3 ttb_gidx = ttb.group_index();
	unsigned int j = ttb_gidx.x + ttb_gidx.y * gridDim.x; 

	// if j is even, 0, 2, ... < N_x*N_y
	if ((j % 2) ==0) 
	{
		Sysparam spinflipresults = spinflips(ttb, Sptr, transprob, temp, Lx,Ly,J, state);
	
		atomicAdd(&(sysparams->E), spinflipresults.E ); 
		atomicAdd(&(sysparams->M), spinflipresults.M ); 
	}
	if ((j % 2) != 0) {
		Sysparam spinflipresults = spinflips(ttb, Sptr, transprob, temp, Lx,Ly,J, state);
	
		atomicAdd(&(sysparams->E), spinflipresults.E ); 
		atomicAdd(&(sysparams->M), spinflipresults.M ); 
	}

//	ttb

};


/**
 * @fn metropolis
 * @brief "driver" function for Metropolis algorithm, single-spin flip scheme for 2-dim. Ising model  
 * */
void metropolis(Spins2d& spins2d, Sysparam_ptr& sysParams,TransProb_ptr& transProbs,
	const std::array<int,3> MAXGRIDSIZES,const dim3 M_is, devStatesXOR & devStates, const unsigned int trials) {

	size_t Lx = spins2d.L_is[0]; // total number of spins of system
	size_t Ly = spins2d.L_is[1]; // total number of spins of system
	const float J = spins2d.J; 

	unsigned int RAD = 1; // "radius" or width of "halo" cells needed 

	/* ========== (thread) grid,block dims ========== */ 
	unsigned long MAX_BLOCKS_y = (MAXGRIDSIZES[1] + M_is.y - 1)/ M_is.y; 
	unsigned int N_y = std::min( MAX_BLOCKS_y, ((Ly + M_is.y - 1)/ M_is.y)); 
	// notice how we're only launching 1/4 of Ly threads in y-direction needed
	unsigned int N_y_4th = std::min( MAX_BLOCKS_y, ((Ly/4 + M_is.y - 1)/ M_is.y)); 
	unsigned long MAX_BLOCKS_x = (MAXGRIDSIZES[0] + M_is.x - 1)/ M_is.x; 
	unsigned int N_x = std::min( MAX_BLOCKS_x, ((Lx + M_is.x - 1)/ M_is.x)); 
	// notice how we're only launching 1/4 of Lx threads in x-direction needed
	unsigned int N_x_4th = std::min( MAX_BLOCKS_x, ((Lx/4 + M_is.x - 1)/ M_is.x)); 
	dim3 N_is { N_x,N_y };
	dim3 N_is_4th { N_x_4th,N_y_4th }; // single (thread) block dims., i.e. number of threads in a single (thread) block
	int sharedBytes = (M_is.x+2*RAD)*(M_is.y + 2*RAD)* sizeof(int);
	
	/* ========== END of (thread) grid,block dims ========== */ 

	metropolis_kernel<<< N_is,M_is,sharedBytes>>>( spins2d.S.get(), sysParams.d_sysparams.get(), 
		(transProbs.d_transProb.get()->transProb).data(), Lx,Ly, J, devStates.devStates.get()); 
	

	// sanity check
/*	std::array<float,17> h_transProb_out;
	cudaMemcpy(&h_transProb_out, (transProbs.d_transProb.get()->transProb).data(), 17*sizeof(float), 
		cudaMemcpyDeviceToHost);  // possible error have to be of same type
	
	for (unsigned int idx=0; idx<17; idx++) { std::cout << 
			h_transProb_out[idx]
		<< " "; } 
	std::cout << std::endl;
*/	
	

}

namespace cg = cooperative_groups;  // this should go with metropolis.h, initialize_allup_kernel

int main(int argc, char* argv[]) {

	constexpr const float initial_temp = 1.f;  // typically 1.
	constexpr const float final_temp = 3.f;  // typically 3.
	constexpr const float tempstep = 0.05f;  // typically 0.05

	// number of spins, related to 2-dim. grid size Lx x Ly
	std::array<size_t, 2> L_is { 1<<10, 1<<10 }; // 1<<10 = 1024 
	std::array<size_t, 2> L_is_small { 4, 4 }; 
	
	Spins2d spins = {L_is};  
	Spins2d spins_small = {L_is_small};  

	std::cout << " L : " << spins.L_is[0]*spins.L_is[1] << std::endl; 

	Sysparam_ptr sysparams_ptr = { initial_temp } ;
	TransProb_ptr transprob_ptr = { initial_temp , 1.f } ;
	Avg_ptr avgs_ptr;  

//	cudaMemcpyToSymbol(constTransProb,&(transprob_ptr.d_transProb), 1*sizeof(TransProb),0,cudaMemcpyDeviceToDevice);
//	cudaMemcpy(&constTransProb, transprob_ptr.d_transProb.get(),1*sizeof(TransProb),cudaMemcpyDeviceToDevice);

	/* sanity check */
	TransProb h_TransProb_out ;  
	cudaMemcpy(&h_TransProb_out, transprob_ptr.d_transProb.get(), 1*sizeof(TransProb), cudaMemcpyDeviceToHost);  // possible error have to be of same type
//	cudaMemcpyFromSymbol(&h_TransProb_out, constTransProb, 1*sizeof(TransProb));  // possible error have to be of same type
	for (unsigned int idx=0; idx<17; idx++) { std::cout << h_TransProb_out.transProb[idx] << " "; } 
	std::cout << std::endl;


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

	// (thread) grid, block dims for curandstates and other 1-dim. arrays  
	unsigned int M_x = 1<<8; // 2^8 = 256 
	unsigned long MAX_BLOCKS = (MAXGRIDSIZE + M_x - 1)/ M_x; 
	unsigned int N_x = min( MAX_BLOCKS, (( spins.L_is[0]*spins.L_is[1] + M_x - 1)/ M_x)); 

	/* ***** END of (thread) grid,block dims ***** */ 

	initialize_allup(spins,sysparams_ptr, MAXGRIDSIZES, M_is);

	/* sanity check */
	Sysparam h_sysparams_out ;  
	cudaMemcpy(&h_sysparams_out, sysparams_ptr.d_sysparams.get(), 1*sizeof(Sysparam), cudaMemcpyDeviceToHost);  // possible error have to be of same type
	std::cout << " h_sysparams_out : " << h_sysparams_out.E << " " << h_sysparams_out.M << " " 
		<< h_sysparams_out.T << std::endl; 

	// since curand_init calls are slow, do it once for the grid from the host main code
	devStatesXOR devstatesXOR = { spins.L_is[0]*spins.L_is[1], N_x,M_x }; 
	
	metropolis(spins,sysparams_ptr,transprob_ptr,MAXGRIDSIZES,M_is,devstatesXOR,1); 
	
	cudaMemcpy(&h_sysparams_out, sysparams_ptr.d_sysparams.get(), 1*sizeof(Sysparam), cudaMemcpyDeviceToHost);  // possible error have to be of same type
	std::cout << " h_sysparams_out : " << h_sysparams_out.E << " " << h_sysparams_out.M << " " 
		<< h_sysparams_out.T << std::endl; 

	
	
}
