/**
 * @file   : 036sgemm.cu
 * @brief  : cublasSgemm - matrix-matrix multiplication 
 * 				C = \alpha op(A) op(B) + \beta C  
 * @author : Ernest Yeung	ernestyalumni@gmail.com
 * @date   : 20170628  
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
// nvcc -std=c++11 -arch='sm_52' 036sgemm.cu -lcublas -o 028strsv.exe

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#define IDX2C(i,j,ld) (((j)*(ld))+( i ))
#define m 6 									// a - mxk matrix
#define n 4 									// b - kxn matrix
#define k 5 									// c - mxn matrix
int main(void) {
	cudaError_t cudaStat; 					// cudaMalloc status
	cublasStatus_t stat; 			// CUBLAS functions status
	cublasHandle_t handle;						// CUBLAS context
	int i,j;						// i-row index,j-column index
	float* a;							// mxk matrix a on the host
	float* b;							// kxn matrix b on the host
	float* c; 							// mxn matrix c on the host
	a=(float*)malloc(m*k*sizeof(float));		// host memory for a
	b=(float*)malloc(k*n*sizeof(float));		// host memory for b
	c=(float*)malloc(m*n*sizeof(float));		// host memory for c
	// define an mxk matrix a column by column
	int ind=11;										// a:
	for (j=0;j<k;j++){								// 11,17,23,29,35
		for (i=0;i<m;i++){							// 12,18,24,30,36
			a[IDX2C(i,j,m)]=(float)ind++;			// 13,19,25,31,37
		}											// 14,20,26,32,38
	}												// 15,21,27,33,39
													// 16,22,28,34,40
													
	// print a row by row
	printf("a:\n");
		for (i=0;i<m;i++) {
			for (j=0;j<k;j++){
				printf("%5.0f",a[IDX2C(i,j,m)]);
			}
		printf("\n");
	}
	// define a kxn matrix b column by column
	ind=11;												// b:
	for (j=0;j<n;j++) {									// 11,16,21,26
		for (i=0;i<k;i++) {								// 12,17,22,27
			b[IDX2C(i,j,k)]=(float)ind++;				// 13,18,23,28
		}												// 14,19,24,29
	}													// 15,20,25,30
	// print b row by row
	printf("b: \n");
	for(i=0;i<k;i++){
		for (j=0;j<n;j++) {
			printf("%5.0f",b[IDX2C(i,j,k)]);
		}
		printf("\n");
	}
	// define an mxn matrix c column by column
	ind =11;												// c:
	for (j=0;j<n;j++){									// 11,17,23,29
		for (i=0;i<m;i++){								// 12,18,24,30
			c[IDX2C(i,j,m)]=(float)ind++;				// 13,19,25,31
		}												// 14,20,26,32
	}													// 15,21,27,33
														// 16,22,28,34
	// print c row by row
	printf("c:\n");
		for (i=0;i<m;i++) {
			for (j=0;j<n;j++) {
				printf("%5.0f",c[IDX2C(i,j,m)]);
			}
			printf("\n");
		}
	
	// on the device
	float* d_a; 							// d_a - a on the device
	float* d_b; 							// d_b - b on the device
	float* d_c;								// d_c - c on the device
	cudaStat=cudaMalloc((void**)&d_a,m*k*sizeof(*a));	// device
												// memory alloc for a
	cudaStat=cudaMalloc((void**)&d_b,k*n*sizeof(*b)); 	// device
												// memory alloc for b
	cudaStat=cudaMalloc((void**)&d_c,m*n*sizeof(*c));	// device
												// memory alloc for c
	stat = cublasCreate(&handle); 		// initialize CUBLAS context
	// copy matrices from the host to the device
	stat = cublasSetMatrix(m,k,sizeof(*a),a,m,d_a,m); 	// a -> d_a
	stat = cublasSetMatrix(k,n,sizeof(*b),b,k,d_b,k);	// b -> d_b
	stat = cublasSetMatrix(m,n,sizeof(*c),c,m,d_c,m);	// c -> d_c
	float a1=1.0f;											// a1=1
	float bet=1.0f;											// bet=1
	// matrix-matrix multiplication: d_c = a1*d_a*d_b + bet*d_c
	// d_a -mxk matrix, d_b - kxn matrix, d_c -mxn matrix;
	// a1,bet - scalars
	
	stat=cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,&a1,d_a,m,d_b,k,&bet,d_c,m);
	
	stat=cublasGetMatrix(m,n,sizeof(*c),d_c,m,c,m);	// cp d_c -> c
	printf("c after Sgemm :\n");
	for(i=0;i<m;i++){
		for (j=0;j<n;j++){
			printf("%7.0f", c[IDX2C(i,j,m)]); 	// print c after Sgemm
		}
		printf("\n");
	}
	cudaFree(d_a);								// free device memory
	cudaFree(d_b);								// free device memory
	cudaFree(d_c);								// free device memory
	cublasDestroy(handle);					// destroy CUBLAS context
	free(a);									// free host memory
	free(b);									// free host memory
	free(c);									// free host memory
	return EXIT_SUCCESS;
}

// a:
//   11   17   23   29   35
//   12   18   24   30   36
//   13   19   25   31   37
//   14   20   26   32   38
//   15   21   27   33   39
//   16   22   28   34   40
// b: 
//   11   16   21   26
//   12   17   22   27
//   13   18   23   28
//   14   19   24   29
//   15   20   25   30
// c:
//   11   17   23   29
//   12   18   24   30
//   13   19   25   31
//   14   20   26   32
//   15   21   27   33
//   16   22   28   34
// c after Sgemm :
//   1566   2147   2728   3309
//   1632   2238   2844   3450  	// c=a1*a*b+bet*c
//   1698   2329   2960   3591
//   1764   2420   3076   3732
//   1830   2511   3192   3873
//   1896   2602   3308   4014


												
												
												
											
