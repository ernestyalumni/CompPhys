/**
 * @file   : XORMRGgens.h
 * @brief  : Examples using cuRAND device API to generate pseudorandom numbers using either XORWOW or MRG32k3a generators, header file   
 * @details : This program uses the device CURAND API.  The purpose of these examples is explore scope and compiling and modularity/separation issues with CURAND   
 * 
 * @author : Ernest Yeung <ernestyalumni@gmail.com>
 * @date   : 20180101      
 * @ref    : http://docs.nvidia.com/cuda/curand/device-api-overview.html#device-api-example
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
 * nvcc -lcurand -dc XORMRGgens2distri.cu -o XORMRGgens2distri  
 * */
 
#ifndef __XORMRGGENS_H__
#define __XORMRGGENS_H__  

#include <curand_kernel.h>  

#include <memory> // std::unique_ptr


#endif // END of __XORMRGGENS_H__
