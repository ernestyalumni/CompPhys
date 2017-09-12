/**
 * @file   : defineGridBlock.cu
 * @brief  : Define grid and block dimensions on the host  
 * @details : Illustrated that when block size is altered, the grid size 
 * 				will be changed accordingly.  
 * @author : Ernest Yeung	ernestyalumni@gmail.com
 * @date   : 20170911
 * @ref    : cf. John Cheng, Max Grossman, Ty McKercher.  Professional CUDA Programming.  
 * @note	: cudaDeviceReset() will explicitly destroy and clean up
 * 				all resources associated iwth the current device in the 
 * 				current process.  
 * 
 * If you find this code useful, feel free to donate directly and easily at this direct PayPal link: 
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
 * P.S. I'm using an EVGA GeForce GTX 980 Ti which has at most compute capability 5.2; 
 * A hardware donation or financial contribution would help with obtaining a 1080 Ti with compute capability 6.2
 * so that I can do -arch='sm_62' and use many new CUDA Unified Memory features
 * */
 /**
  * COMPILATION TIP(s)
  * (without make file, stand-alone)
  * nvcc checkDimension.cu -o checkDimension.exe
  * 
  * */
#include <cuda_runtime.h>
#include <stdio.h>


int main(int argc, char **argv) {
	// define total data elements
	int nElem = 1024; 
	
	// define grid and block structure
	dim3 block (1024);
	dim3 grid ((nElem+block.x -1)/block.x);
	printf("grid.x %d block.x %d \n",grid.x, block.x);

	// reset block
	block.x = 512; 
	grid.x = (nElem+block.x-1)/block.x;
	printf("grid.x %d block.x %d \n", grid.x, block.x);
	
	// reset block
	block.x = 256; 
	grid.x = (nElem+block.x-1)/block.x;
	printf("grid.x %d block.x %d \n", grid.x, block.x);

	// reset block
	block.x = 128; 
	grid.x = (nElem+block.x-1)/block.x;
	printf("grid.x %d block.x %d \n", grid.x, block.x);
	
	// reset device before you leave
	cudaDeviceReset(); 
	return(0); 
}
