/** \file 001isamax_unified.cu
 * \author Ernest Yeung
 * \email  ernestyalumni@gmail.com
 * \brief This function performs matrix-vector multiplication
 *  		y = \alpha op(A) x + \beta y
 * 3.3.2 	cublasSgemv - matrix-vector multiplication 
 * cf. https://developer.nvidia.com/sites/default/files/akamai/cuda/files/Misc/mygpu.pdf
 * 
 * If you find this code useful, feel free to donate directly and easily at this direct PayPal link: 
 * 
 * https://www.paypal.com/cgi-bin/webscr?cmd=_donations&business=ernestsaveschristmas%2bpaypal%40gmail%2ecom&lc=US&item_name=ernestyalumni&currency_code=USD&bn=PP%2dDonationsBF%3abtn_donateCC_LG%2egif%3aNonHosted 
 * 
 * which won't go through a 3rd. party such as indiegogo, kickstarter, patreon.  
 * Otherwise, I receive emails and messages on how all my (free) material on physics, math, and engineering have 
 * helped students with their studies, and I know what it's like to not have money as a student, but love physics 
 * (or math, sciences, etc.), so I am committed to keeping all my material open-source and free, whether or not 
 * sufficiently crowdfunded, under the open-source MIT license: 
 * 	feel free to copy, edit, paste, make your own versions, share, use as you wish.    
 * Peace out, never give up! -EY
 * 
 * */
 /** 1.2. New and Legacy cuBLAS API
  * cf. http://docs.nvidia.com/cuda/cublas/index.html#new-and-legacy-cublas-api
  * 
  * */
 /**
 * Compilation tips
 *
 * nvcc -lcublas 001isamax.cu -o 001isamax.exe 
 * 
 **/
 // COMPILATION TIP:
// nvcc -std=c++11 -arch='sm_61' 001isamax_unified.cu -lcublas -o 001isamax_unified.exe

#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

constexpr int m {6};					// number of rows of a 
constexpr int n {5}; 				// number of columns of a 

__device__ __managed__ float a[m*n];  	// a - m x n matrix on the managed device
__device__ __managed__ float x[n];  	// x - n-vector on the managed device
__device__ __managed__ float y[m];  	// y - m-vector on the managed device


int main(void) {
	cudaError_t cudaStat;			// cudaMalloc status
	cublasStatus_t stat;  		// CUBLAS functions status
	cublasHandle_t handle; 				// CUBLAS context

	// cf. https://stackoverflow.com/questions/12400477/retaining-dot-product-on-gpgpu-using-cublas-routine/12401838#12401838
	cublasCreate(&handle);
//	cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);

	int i,j; 				// i-row index, j-column index
	
	// define an m x n matrix a - column by column
	int ind=11;  							// a:
	for (j=0;j<n;j++) {						// 11,17,23,29,35
		for (i=0;i<m;i++) {					// 12,18,24,30,36
			a[ m*j + i ] = (float)ind++;	// 13,19,25,31,37
		}
	}
	
	std::cout << "a:" << std::endl; 
	for (i=0;i<m;i++) {
		for (j=0;j<n;j++) {
			std::cout << a[j*m +i]<< " "; // print a row by row
		}
		std::cout << std::endl; 
	}
	
	for (i=0; i<n;i++) { x[i] = 1.0f; } 	// x={1,1,1,1,1}^T
	for (i=0; i<m;i++) { y[i] = 0.0f; } 	// x={0,0,0,0,0,0}^T
	
	float a1=1.0f; 		// a1 =1
	float bet=1.0f; 	// bet=1

	// matrix-vector multiplication: y = a1*a*x + bet*y
	// a - m x n matrix; x - n-vector; y - m-vector ;
	// a1,bet - scalars
	
	stat=cublasSgemv(handle,CUBLAS_OP_N,m,n,&a1,a,m,x,1,&bet,y,1);

	cudaDeviceSynchronize(); 
	
	std::cout<<"y after Sgemv::\n" ; 
	for(j=0;j<m;j++) {
		std::cout << y[j] << std::endl; // print y after Sgemv
	}
	
	return EXIT_SUCCESS;


}
// a :
//	11 	17 	23 	29 	35
// 	12 	18 	24 	30 	36
	
	
