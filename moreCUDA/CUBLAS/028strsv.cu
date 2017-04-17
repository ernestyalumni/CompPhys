/**
 * @file   : 028strsv.cu
 * @brief  : cublasStbsv - solve the triangular linear system  
 * uses CUDA Unified Memory (Management)
 * @author : Ernest Yeung	ernestyalumni@gmail.com
 * @date   : 20170417
 * @ref    :  cf. https://developer.nvidia.com/sites/default/files/akamai/cuda/files/Misc/mygpu.pdf
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
 * */
// COMPILATION TIP:
// nvcc -std=c++11 -arch='sm_52' 028strsv.cu -lcublas -o 028strsv.exe
#include <iostream>
#include "cublas_v2.h"

constexpr const int n =6;			// number of rows and columns of A 

__device__ __managed__ float A[n*n];	// nxn matrix a on CUDA Unified (managed) memory 
__device__ __managed__ float b[n]; 		// n-vector b on CUDA Unified (managed) memory 
__device__ __managed__ float x[n]; 		// n-vector x on CUDA Unified (managed) memory 


int main(void){
	cudaError_t cudaStat; 						// cudaMalloc status
	cublasStatus_t stat;					// CUBLAS functions status
	cublasHandle_t handle;						// CUBLAS context
	int i,j;							// i-row index, j-column index
	
	// column by column
	int ind=11;							// A:
	for (j=0;j<n;j++) {					// 11
		for (i=0;i<n;i++) {				// 12,17
			if (i>=j) {					// 13,18,22
				A[i+n*j]=(float)ind++;	// 14,19,23,26
			}							// 15.20.24,27,29
		}								// 16,21,25,28,30,31
	}
	for (i=0;i<n;i++) {
		b[i]=1.0f; 						// x={1,1,1,1,1,1}^T
	}

	stat = cublasCreate(&handle);		// initialize CUBLAS context
	
	// solve the triangular linear system: A*x = b
	// the solution x overwrite the right hand side b,
	// A - nxn triangular matrix in lower mode; b - n-vector
	
	stat=cublasStrsv(handle,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_N,
							CUBLAS_DIAG_NON_UNIT,n,A,n,b,1);
							
	stat=cublasGetVector(n,sizeof(float),b,1,x,1);	// copy b->x

							
	printf("solution :\n");				// print x after Strsv
	for (j=0;j<n;j++) {
		std::cout << b[j] << std::endl;
	}
	
	cublasDestroy(handle);				// destroy CUBLAS context
	
	return EXIT_SUCCESS;
}

	
