/**
 * @file   : checkDimension.cu
 * @brief  : Check grid and block indices and dimensions  
 * @author : Ernest Yeung	ernestyalumni@gmail.com
 * @date   : 20170911
 * @ref    : cf. John Cheng, Max Grossman, Ty McKercher.  Professional CUDA Programming.  
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

__global__ void checkIndex(void) {
	printf("threadIdx:(%d, %d, %d) blockIdx:(%d, %d, %d) blockDim:(%d, %d, %d) " 
		"gridDim: (%d, %d, %d) \n", threadIdx.x, threadIdx.y, threadIdx.z, 
		blockIdx.x, blockIdx.y, blockIdx.z, blockDim.x, blockDim.y, blockDim.z,
		gridDim.x, gridDim.y, gridDim.z
		);
}

int main(int argc, char **argv) {
	// define total data element
	int nElem = 6; 
	
	// define grid and block structure
	dim3 block (3);
	dim3 grid ((nElem+block.x -1)/block.x);
	
	// check grid and block dimension from host side
	printf("grid.x %d block.y %d block.z %d\n", block.x, block.y, block.z);
	printf("block.x %d block.y %d block.z %d\n", block.x, block.y, block.z);
	
	// check grid and block dimension from device side
	checkIndex<<<grid, block>>>();
	
	// reset device before you leave
	cudaDeviceReset();
	
	return(0);
}
  
