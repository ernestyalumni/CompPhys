/**
 * @file   : metropolis.h
 * @brief  : Metropolis algorithm for 2-dim. grid, with initialization, header file, in CUDA C++11/14, 
 * @details : initialize function, metropolis function
 * 
 * @author : Ernest Yeung <ernestyalumni@gmail.com>
 * @date   : 20170110    
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
 * nvcc main.cu ./grid2d/grid2d.cu ./grid2d/sysparam.cu ./dynam/XORMRGgens.cu -o main
 * 
 * */
#ifndef __METROPOLIS_H__  
#define __METROPOLIS_H__  

#include "../grid2d/grid2d.h" // Spins2d struct 
#include "../grid2d/sysparam.h" // Sysparam_ptr struct 
//#include "../boundary/boundary.h" // periodic boundary conditions as inline __device__ function periodic // ptxas fatal compile error: 
#include "../dynam/XORMRGgens.h" // devStatesXOR, devStatesMRG, devStatesPhilox4_32_10_t

#include <algorithm> // std::min

#include <cooperative_groups.h>  // this should go with metropolis.h, initialize_allup_kernel

namespace cg = cooperative_groups;  // this should go with metropolis.h, initialize_allup_kernel

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
__device__ int periodic_nn(const int i, const int L, const int nu) ; 

/**
 * @fn periodic
 * @brief periodic boundary conditions; Choose correct matrix index with 
 * periodic boundary conditions 
 * 
 * Input :
 * @param - i 		: Base index 
 * @param - L 	: Highest \"legal\" index
 */
__device__ int periodic(const int i, const int L) ; 


/* =============== END of boundary conditions =============== */

/* =============== Initialization =============== */

/** @fn init_allup_partialsumM
 * @brief initialize spins all up and calculate partial sums for magnetization M
 * @details 1st part of initialize_allup_kernel, 2nd. part is block_sumM
 * */
__device__ int init_allup_partialsumM(int* Sptr,size_t Lx,size_t Ly) ;

/** @fn blocksumM
 * @brief reduce sum on thread block of partial sums of spins for magnetization M 
 * @details 2nd. part of initialize_allup_kernel, 1st. part is init_allup_partialsumM 
 * */
__device__ int block_sumM(cg::thread_group tg, int* temp, int sumresult);

/** @fn calcE
 * @brief computes E, a summation of all unique nearest neighbor pairs of spins 
 * @details do summation in shared memory, that include halo cells of width 1 "to the right"
 * */
__device__ int calcE(cg::thread_group & tg, int* Sptr, int* temp, size_t Lx, size_t Ly, const float J);

__global__ void initialize_allup_kernel(int* Sptr, Sysparam* sysparams, size_t Lx, size_t Ly, const float J) ;

__global__ void calcE_kernel(int* Sptr, Sysparam* sysparams, size_t Lx, size_t Ly, const float J) ;

/**
 * @fn initialize_allup
 * @brief "driver" function to initialize energy, spin matrix, and magnetization 
 * */
void initialize_allup(Spins2d& spins2d, Sysparam_ptr& sysParams,
	const std::array<int,3> MAXGRIDSIZES,const dim3 M_is={32,32});

/* =============== END of initialization =============== */

/* =============== Metropolis algorithm =============== */

__device__ int calcintDeltaE(int* temp, const unsigned int S_x, const unsigned int S_y, 
	const unsigned int s_x, const unsigned int s_y, const int RAD) ; 

__device__ Sysparam spinflips(cg::thread_group & tg, int* Sptr, float * transprob, 
								int* temp, size_t Lx, size_t Ly, const float J, curandState *state) ;

__global__ void metropolis_kernel(int* Sptr, Sysparam* sysparams,float* transprob, size_t Lx, size_t Ly, 
									const float J, curandState *state); 

__global__ void update_avgs(Sysparam* sysparams,Avg* avgs);


/**
 * @fn metropolis
 * @brief "driver" function for Metropolis algorithm, single-spin flip scheme for 2-dim. Ising model  
 * */
void metropolis(Spins2d& spins2d, Sysparam_ptr& sysParams,Avg_ptr& averages,TransProb_ptr& transProbs,
	const std::array<int,3> MAXGRIDSIZES,const dim3 M_is, devStatesXOR & devStates, const unsigned int trials); 


/* =============== END of Metropolis algorithm =============== */


#endif // END of __METROPOLIS_H__
