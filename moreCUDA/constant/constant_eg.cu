/**
 * @file   : constant_eg.cu
 * @brief  : Examples of using constant memory for CUDA 
 * @details : constant memory for CUDA examples
 *  
 * @author : Ernest Yeung <ernestyalumni@gmail.com>
 * @date   : 20170103      
 * @ref    : http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-specifiers
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
 * nvcc constant_eg.cu -o constant_eg
 * 
 * */
#include <iostream>   

__constant__ float constData_global[256]; 
__device__ float devData;  
__device__ float* devPointer; 


int main(int argc, char* argv[]) {
	float data_main[256]; 

	/* "boilerplate" test values */ 
	for (int idx=0; idx<256; idx++) { 
		data_main[idx] = ((float) idx+1);  
	}
	
	cudaMemcpyToSymbol(constData_global, data_main, sizeof(data_main)); 
	
	float data_main1[256]; 
	for (int idx=0; idx < 256; idx++) { std::cout << data_main1[idx] << " "; }

	cudaMemcpyFromSymbol(data_main1, constData_global, sizeof(data_main1) );
	/* sanity check */ 
	for (int idx=0; idx < 256; idx++) { std::cout << data_main1[idx] << " "; }
	
//	__constant__ float constData_main[256]; // error:  a "__constant__" 
	// variable declaration is not allowed inside a function body

//	__device__ float devData; // error: a "__device__" variable declaration is not allowed inside a function body
	float value = 3.14; 
	cudaMemcpyToSymbol(devData, &value, sizeof(float)); 
	
	float *ptr; 
	cudaMalloc(&ptr, 256*sizeof(float)); 
	cudaMemcpyToSymbol(devPointer, &ptr, sizeof(ptr));  
	
	

}
