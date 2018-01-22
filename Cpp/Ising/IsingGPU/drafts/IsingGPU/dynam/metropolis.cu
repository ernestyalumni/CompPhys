/**
 * @file   : metropolis.cu
 * @brief  : Metropolis algorithm for 2-dim. grid, with initialization, separate/implementation file, in CUDA++11/14, 
 * @details : initialize function, metropolis function
 * 
 * @author : Ernest Yeung <ernestyalumni@gmail.com>
 * @date   : 20180110    
 * @ref    : M. Hjorth-Jensen, Computational Physics, University of Oslo (2015) 
 * 
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
 * g++ main.cpp ./structs/structs.cpp -o main
 * 
 * */
#include "./metropolis.h"  

/* =============== boundary conditions =============== */

/**
 * @fn periodic_nn
 * @brief periodic boundary conditions; Choose correct matrix index with 
 * periodic boundary conditions 
 * 
 * Input :
 * @param - i 		: Base index 
 * @param - L 	: Highest \"legal\" index
 * @param - nu		: Number to add or subtract from i
 */
__device__ int periodic_nn(const int i, const int L, const int nu) {
	return ( i + nu ) % L; // (i + nu) = 0,1,...L-1 
} 

/**
 * @fn periodic
 * @brief periodic boundary conditions; Choose correct matrix index with 
 * periodic boundary conditions 
 * 
 * Input :
 * @param - i 		: Base index 
 * @param - L 	: Highest \"legal\" index
 */
__device__ int periodic(const int i, const int L) {
	return i % L; // i  = 0,1,...L-1 
} 

/* =============== END of boundary conditions =============== */


/* =============== Initialization =============== */

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
} 

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
}

/** @fn calcE
 * @brief computes E, a summation of all unique nearest neighbor pairs of spins 
 * @details do summation in shared memory, that include halo cells of width 1 "to the right".  
 * lx+idxx*gridDim.x*blockDim.x, idxx=0,1,.. to how many multiples of gridDim.x*blockDim.x for 
 * multiples of thread grids to "cover" our lattice grid of spins.  
 * (lx+idxx*gridDim.x*blockDim.x)%Lx because we want periodic boundary conditions  
 * I try to future proof this by using inline function periodic
 * */
__device__ int calcE(cg::thread_group & tg, int* Sptr, int* temp, size_t Lx, size_t Ly, const float J) {
	int resultE =0;

	const int RAD = 1; // "radius" of "halo" cells, of width 1 (in this case)  

	// old way of thread, block indexing
	unsigned int k_x = threadIdx.x + blockDim.x * blockIdx.x ; 
	unsigned int k_y = threadIdx.y + blockDim.y * blockIdx.y ;
	unsigned int kidx = k_x + k_y * gridDim.x * blockDim.x  ;

	unsigned int S_x = static_cast<int>(blockDim.x + RAD); 
	unsigned int S_y = static_cast<int>(blockDim.y + RAD); 
	
	unsigned int s_x = threadIdx.x + RAD; // s_x = 1,2,...S_x-1
	unsigned int s_y = threadIdx.y + RAD; // s_y = 1,2,...S_y-1


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
	// partial sum of spins for magnetization M
	int sum4M = init_allup_partialsumM( Sptr, Lx,Ly); 
	extern __shared__ int temp[]; 
	auto ttb = cg::this_thread_block(); 
	int block_sum = block_sumM(ttb, temp, sum4M) ;

	if (ttb.thread_rank() == 0) {
		atomicAdd(&(sysparams->M), ((float) block_sum)); 
	}

//	int threadsumE = calcE(ttb, Sptr, temp, Lx,Ly,J); // for this thread, here's its partial sum contribution to total energy E
//	atomicAdd(&(sysparams->E), ((float) threadsumE) ); 
}

__global__ void calcE_kernel(int* Sptr, Sysparam* sysparams, size_t Lx, size_t Ly, const float J) {
	extern __shared__ int temp[]; 
	auto ttb = cg::this_thread_block(); 
	int threadsumE = calcE(ttb, Sptr, temp, Lx,Ly,J); // for this thread, here's its partial sum contribution to total energy E
	atomicAdd(&(sysparams->E), ((float) threadsumE) ); 
}

/**
 * @fn initialize_allup
 * @brief "driver" function to initialize energy, spin matrix, and magnetization 
 * */
void initialize_allup(Spins2d& spins2d, Sysparam_ptr& sysParams,
	const std::array<int,3> MAXGRIDSIZES,const dim3 M_is) 
{ 
	size_t Lx = spins2d.L_is[0]; // total number of spins of system
	size_t Ly = spins2d.L_is[1]; // total number of spins of system
	const float J = spins2d.J; 

	unsigned int RAD = 1; // "radius" or width of "halo" cells needed 

	/* ========== (thread) grid,block dims ========== */ 
	unsigned long MAX_BLOCKS_y = (MAXGRIDSIZES[1] + M_is.y - 1)/ M_is.y; 
	// notice how we're only launching 1/4 of Ly threads in y-direction needed
	unsigned int N_y = std::min( MAX_BLOCKS_y, ((Ly/4 + M_is.y - 1)/ M_is.y)); 
	unsigned int N_y_full = std::min( MAX_BLOCKS_y, ((Ly + M_is.y - 1)/ M_is.y)); 
	unsigned long MAX_BLOCKS_x = (MAXGRIDSIZES[0] + M_is.x - 1)/ M_is.x; 
	// notice how we're only launching 1/4 of Lx threads in x-direction needed
	unsigned int N_x = std::min( MAX_BLOCKS_x, ((Lx/4 + M_is.x - 1)/ M_is.x)); 
	unsigned int N_x_full = std::min( MAX_BLOCKS_x, ((Lx + M_is.x - 1)/ M_is.x)); 
	dim3 N_is { N_x,N_y }; // single (thread) block dims., i.e. number of threads in a single (thread) block
	dim3 N_is_full { N_x_full,N_y_full }; // single (thread) block dims., i.e. number of threads in a single (thread) block
	int sharedBytes = (M_is.x+RAD)*(M_is.y + RAD)* sizeof(int);
	
	/* ========== END of (thread) grid,block dims ========== */ 

	initialize_allup_kernel<<<N_is,M_is, sharedBytes>>>(spins2d.S.get(),sysParams.d_sysparams.get(),Lx,Ly,J);
	calcE_kernel<<<N_is_full,M_is,sharedBytes>>>(spins2d.S.get(),sysParams.d_sysparams.get(),Lx,Ly,J); 

} // end of function initialize_allup 

__device__ int unifl2intspin(const float unif) {
	return (2 * static_cast<int>(floorf(2.f*unif)) - 1); 
}

/** @fn init_rand_partialsumM
 * @brief initialize spins all up and calculate partial sums for magnetization M
 * @details 1st part of initialize_allup_kernel, 2nd. part is block_sumM
 * */
__device__ int init_rand_partialsumM(int* Sptr,size_t Lx,size_t Ly,curandState *state) {
	int sum=0; // partial sum of the magnetization M

	// global thread index, k_x = 0,1,...N_x*M_x, k_y = 0,1,...N_y*M_y
	unsigned int k_x = threadIdx.x + blockDim.x * blockIdx.x ; 
	unsigned int k_y = threadIdx.y + blockDim.y * blockIdx.y ; 
	unsigned int k = k_x + gridDim.x * blockDim.x * k_y;  

	curandState localState = state[k]; 

	for (unsigned int idx = k; idx < Lx*Ly/4; idx+= blockDim.x * gridDim.x * blockDim.y * gridDim.y ) {
		float ranf = curand_uniform(&localState); 
		int ranint0 = unifl2intspin(ranf); 
		ranf = curand_uniform(&localState); 
		int ranint1 = unifl2intspin(ranf); 
		ranf = curand_uniform(&localState); 
		int ranint2 = unifl2intspin(ranf); 
		ranf = curand_uniform(&localState); 
		int ranint3 = unifl2intspin(ranf); 

		reinterpret_cast<int4*>(Sptr)[idx] = {ranint0,ranint1,ranint2,ranint3} 	;
		int4 s4 = ((int4*) Sptr)[idx]; 
		sum += s4.x + s4.y + s4.z + s4.w; 
	}
	
	// process remaining elements
	for (unsigned int idx = k + Lx*Ly/4 *4; idx < Lx*Ly; idx += 4) { 
		float ranf = curand_uniform(&localState); 
		int ranint = unifl2intspin(ranf); 

		Sptr[idx] = ranint; 
		sum += Sptr[idx]; 
	}
	return sum;
} 

__global__ void initialize_rand_kernel(int* Sptr, Sysparam* sysparams, size_t Lx, size_t Ly, 
	curandState *state) {
	// global thread index, k_x = 0,1,...N_x*M_x, k_y = 0,1,...N_y*M_y
	// partial sum of spins for magnetization M
	int sum4M = init_rand_partialsumM( Sptr, Lx,Ly, state); 
	extern __shared__ int temp[]; 
	auto ttb = cg::this_thread_block(); 
	int block_sum = block_sumM(ttb, temp, sum4M) ;

	if (ttb.thread_rank() == 0) {
		atomicAdd(&(sysparams->M), ((float) block_sum)); 
	}
}

/**
 * @fn initialize_rand
 * @brief "driver" function to initialize energy, spin matrix, and magnetization 
 * */
void initialize_rand(Spins2d& spins2d, Sysparam_ptr& sysParams,
	const std::array<int,3> MAXGRIDSIZES,devStatesXOR & devStates,const dim3 M_is) 
{ 
	size_t Lx = spins2d.L_is[0]; // total number of spins of system
	size_t Ly = spins2d.L_is[1]; // total number of spins of system
	const float J = spins2d.J; 

	unsigned int RAD = 1; // "radius" or width of "halo" cells needed 

	/* ========== (thread) grid,block dims ========== */ 
	unsigned long MAX_BLOCKS_y = (MAXGRIDSIZES[1] + M_is.y - 1)/ M_is.y; 
	// notice how we're only launching 1/4 of Ly threads in y-direction needed
	unsigned int N_y = std::min( MAX_BLOCKS_y, ((Ly/4 + M_is.y - 1)/ M_is.y)); 
	unsigned int N_y_full = std::min( MAX_BLOCKS_y, ((Ly + M_is.y - 1)/ M_is.y)); 
	unsigned long MAX_BLOCKS_x = (MAXGRIDSIZES[0] + M_is.x - 1)/ M_is.x; 
	// notice how we're only launching 1/4 of Lx threads in x-direction needed
	unsigned int N_x = std::min( MAX_BLOCKS_x, ((Lx/4 + M_is.x - 1)/ M_is.x)); 
	unsigned int N_x_full = std::min( MAX_BLOCKS_x, ((Lx + M_is.x - 1)/ M_is.x)); 
	dim3 N_is { N_x,N_y }; // single (thread) block dims., i.e. number of threads in a single (thread) block
	dim3 N_is_full { N_x_full,N_y_full }; // single (thread) block dims., i.e. number of threads in a single (thread) block
	int sharedBytes = (M_is.x+RAD)*(M_is.y + RAD)* sizeof(int);
	
	/* ========== END of (thread) grid,block dims ========== */ 

	initialize_rand_kernel<<<N_is,M_is, sharedBytes>>>(spins2d.S.get(),sysParams.d_sysparams.get(),Lx,Ly,
		devStates.devStates.get());
	calcE_kernel<<<N_is_full,M_is,sharedBytes>>>(spins2d.S.get(),sysParams.d_sysparams.get(),Lx,Ly,J); 

} // end of function initialize_allup 



/* =============== END of initialization =============== */


/* =============== Metropolis algorithm =============== */

__device__ int calcintDeltaE(int* temp, const unsigned int S_x, const unsigned int S_y, 
	const unsigned int s_x, const unsigned int s_y, const int RAD) 
{ 
	int resultDeltaE = 2 * temp[ s_x + s_y * S_x] * 
				(temp[ s_x + 1 + s_y * S_x]
					+ temp[ s_x + (s_y + 1)*S_x] 
					+ temp[ s_x - 1 + s_y * S_x]
					+ temp[ s_x + (s_y - 1)*S_x] ); 
					
	return resultDeltaE;  
}

__device__ Sysparam spinflips(cg::thread_group & tg, int* Sptr,  
								int* temp, size_t Lx, size_t Ly, const float J, curandState *state, const float T) 
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
//			if ( ( tg.thread_rank() % 2) == 0) 
			if  ((( k_x+k_y ) % 2) == 0 )
//			if ( (k  % 2) == 0 )
			{
				// pick ALL the "even" spins in this thread block
				// do the nearest neighbor (unique) pair of spins summation entirely in shared memory
				int intdeltaE = calcintDeltaE(temp, S_x,S_y,s_x,s_y,RAD);					

				float Wprob = curand_uniform(&localState); 
				float transprob = expf( -1.f / T * ( static_cast<float>( intdeltaE) ) );

				// roll dice, see if we transition or not, given transprob
//				if ( curand_uniform(&localState) <= transprob[intdeltaE +8] ) 
				if (intdeltaE <0 || Wprob <= transprob) 
				{
					// instead of loading entire thread block + halo cells to shared memory again, just make single change  

					// Accept!  
					temp[ s_x + s_y * S_x] *= -1; // flip 1 spin and accept new spin config 
					results_sysparams.E += ((float) intdeltaE) * J ; 
					results_sysparams.M += 2.f*((float) temp[s_x+s_y*S_x]) ; 
				}
			}
			tg.sync(); 
			// if tg.thread_rank() odd
//			if ( (tg.thread_rank() % 2) == 1) 
//			if (( k  % 2) == 1 )
			if (( (k_x + k_y) % 2) == 1 )
			{
				// do the nearest neighbor (unique) pair of spins summation entirely in shared memory
				int intdeltaE = calcintDeltaE(temp, S_x,S_y,s_x,s_y,RAD);					

				float Wprob = curand_uniform(&localState); 
				float transprob = expf( -1.f / T * ( static_cast<float>( intdeltaE) ) );

				// roll dice, see if we transition or not, given transprob
//				if ( curand_uniform(&localState) <= transprob[intdeltaE +8] ) 
				if (intdeltaE <0 || Wprob <= transprob) 
				{
					// Accept!  
					temp[ s_x + s_y * S_x] *= -1; // flip 1 spin and accept new spin config 
					results_sysparams.E += ((float) intdeltaE) * J ; 
					results_sysparams.M += 2.f*((float) temp[s_x+s_y*S_x]) ; 
				}	 
			}
			tg.sync(); 
			// coalesce global memory access with all threads consecutively in memory 
			Sptr[k] = temp[ s_x + s_y * S_x]; 

			tg.sync(); // Added
		
		}
	} // END of loops to make threads do "double duty" to cover other elements in our spin lattice grid that wasn't "covered" by our thread grid
	return results_sysparams ;  
};


__global__ void metropolis_kernel(int* Sptr, Sysparam* sysparams, size_t Lx, size_t Ly, const float J,
									curandState *state ) 
{
	extern __shared__ int temp[]; 
	auto ttb = cg::this_thread_block(); 
	
	dim3 ttb_gidx = ttb.group_index();
//	unsigned int j = ttb_gidx.x + ttb_gidx.y * gridDim.x; 
	unsigned int j = blockIdx.x + gridDim.x * blockIdx.y;

	// if j is even, 0, 2, ... < N_x*N_y
/*	if ((j % 2) ==0) 
	{
//		Sysparam spinflipresults = spinflips(ttb, Sptr, transprob, temp, Lx,Ly,J, state);
		Sysparam spinflipresults = spinflips(ttb, Sptr, temp, Lx,Ly,J, state,sysparams->T);
	
		atomicAdd(&(sysparams->E), spinflipresults.E ); 
		atomicAdd(&(sysparams->M), spinflipresults.M ); 
	}
	ttb.sync();
	
	if ((j % 2) != 0) {
//		Sysparam spinflipresults = spinflips(ttb, Sptr, transprob, temp, Lx,Ly,J, state);
		Sysparam spinflipresults = spinflips(ttb, Sptr, temp, Lx,Ly,J, state,sysparams->T);
	
		atomicAdd(&(sysparams->E), spinflipresults.E ); 
		atomicAdd(&(sysparams->M), spinflipresults.M ); 
	}
	ttb.sync();
*/	
//	Sysparam spinflipresults = spinflips(ttb, Sptr, transprob, temp, Lx,Ly,J, state, sysparams->T );
	Sysparam spinflipresults = spinflips(ttb, Sptr, temp, Lx,Ly,J, state, sysparams->T );
	
	atomicAdd(&(sysparams->E), spinflipresults.E ); 
	atomicAdd(&(sysparams->M), spinflipresults.M ); 

}

__global__ void update_avgs(Sysparam* sysparams,Avg* avgs) {
	auto ttb = cg::this_thread_block(); 
	dim3 ttb_gidx = ttb.group_index(); 
	if ( ((ttb.thread_rank() == 0) && (ttb_gidx.x == 0)) || ( (threadIdx.x == 0)&&(blockIdx.x==0)) ) { 
		atomicAdd(&(avgs->Eavg), sysparams->E); 
		atomicAdd(&(avgs->Mavg), sysparams->M); 
		atomicAdd(&(avgs->Esq_avg), (sysparams->E)*(sysparams->E)); 
		atomicAdd(&(avgs->Msq_avg), (sysparams->M)*(sysparams->M)); 
		atomicAdd(&(avgs->absM_avg), fabsf(sysparams->M)); 

		atomicAdd(&(avgs->M4_avg), (sysparams->M)*(sysparams->M)*(sysparams->M)*(sysparams->M)); 
	}
}


/**
 * @fn metropolis
 * @brief "driver" function for Metropolis algorithm, single-spin flip scheme for 2-dim. Ising model  
 * */
//void metropolis(Spins2d& spins2d, Sysparam_ptr& sysParams,Avg_ptr& averages,TransProb_ptr& transProbs,
void metropolis(Spins2d& spins2d, Sysparam_ptr& sysParams,Avg_ptr& averages,
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

	for (int cycles=1; cycles <= trials; cycles++) { 

		metropolis_kernel<<< N_is,M_is,sharedBytes>>>( spins2d.S.get(), sysParams.d_sysparams.get(), 
			 Lx,Ly, J, 
			devStates.devStates.get() ); 

		update_avgs<<<1,1>>>( sysParams.d_sysparams.get(), averages.d_avgs.get() );  

	}
	

}

/* =============== END of Metropolis algorithm =============== */
