/** \file 001isamax.cu
 * \author Ernest Yeung
 * \email  ernestyalumni@gmail.com
 * \brief This function finds the smallest index of the element of an array with the maximum/minimum magnitude;
 * 3.2.1 cublasIsamax, cublasIsamin - maximal, minimal elements 
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
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"


#define n 6 									// length of x
int main(void) {
	cudaError_t cudaStat; 						// cudaMalloc status
	cublasStatus_t stat; 						// CUBLAS functions status
	cublasHandle_t handle; 						// CUBLAS context
	int j;										// index of elements
	float* x;									// n-vector on the host
	x=(float *)malloc (n*sizeof(*x)); 			// host memory alloc
	for (j=0;j<n;j++)
		x[j]=(float) j;							// x={0,1,2,3,4,5}
	printf("x: ");
	for(j=0;j<n;j++)
		printf("%4.0f,",x[j]);					// print x
	printf("\n");
	// on the device
	float* d_x;									// d_x - x on the device
	cudaStat=cudaMalloc((void**)&d_x,n*sizeof(*x)); 	// device
												// memory alloc for x
	stat = cublasCreate(&handle); 				// initialize CUBLAS context
	stat = cublasSetVector(n,sizeof(*x),x,1,d_x,1); 	// cp x -> d_x
	int result; 								// index of the maximal/minimal element
	// find the smallest index of the element of d_x with maximum
	// absolute value
	
	stat=cublasIsamax(handle,n,d_x,1,&result);
	printf("max |x[i]|:%4.0f\n",fabs(x[result-1])); // print 
												// max{|x[0]|,...,|x[n-1]|}
	// find the smallest index of the element of d_x with minimum
	// absolute value

	stat=cublasIsamin(handle,n,d_x,1,&result);
	
	printf("min |x[i]|:%4.0f\n",fabs(x[result-1])); 			// print
												// min{|x[0]|,...,|x[n-1]|
	cudaFree(d_x);								// free device memory
	cublasDestroy(handle); 						// destroy CUBLAS context
	free(x);									// free host memory
	return EXIT_SUCCESS;
}
// x: 0, 1, 2, 3, 4, 5,
// max |x[i]|:  5
// min |x[i]|:  0
