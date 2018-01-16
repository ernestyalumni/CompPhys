/**
 * @file   : boundary.h
 * @brief  : boundary conditions for 2-dim. Ising model as inline function
 * @details : Choose correct grid index with periodic boundary conditions
 *  
 * @author : Ernest Yeung <ernestyalumni@gmail.com>
 * @date   : 20170112      
 * @ref    : http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#noinline-and-forceinline
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
 * nvcc cg_eg1.cu -o cg_eg1
 * 
 * */

#ifndef __BOUNDARY_H__
#define __BOUNDARY_H__  

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

	
#endif // END of __BOUNDARY_H__
