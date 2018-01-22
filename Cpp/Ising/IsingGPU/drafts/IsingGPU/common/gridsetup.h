/**
 * @file   : gridsetup.h
 * @brief  : Utility functions to set up (thread) grid on device GPU 
 * @details : Utility functions to set up (thread) grid on device GPU   
 *  
 * @author : Ernest Yeung <ernestyalumni@gmail.com>
 * @date   : 20170110      
 * @ref    : https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-increase-performance-with-vectorized-memory-access/
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
 * nvcc vectorloads_eg.cu -o vectorloads_eg
 * DEBUG/ASSEMBLY TIP
 * cuobjdump -sass vectorloads_eg
 * */
#ifndef __GRIDSETUP_H__
#define __GRIDSETUP_H__

#include <array> 		// std::array

/* ********** functions to setup device GPU ********** */

/** @fn getMaxGridSize
 * @brief get maxGridSize (total number threads on a (thread) grid, on device GPU, of a single device GPU
 * */
size_t get_maxGridSize();  

std::array<int,3> get_maxGridSizes();  


#endif // __GRIDSETUP_H__
