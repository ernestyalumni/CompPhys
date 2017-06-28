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

#include <iostream>
#include "cublas_v2.h"

constexpr const int m=6; 								// a - mxk matrix
constexpr const int n=4;								// b - kxn matrix
constexpr const int k=5;								// c - mxn matrix

__device__ __managed__ float a[m*k];				
__device__ __managed__ float b[k*n];
__device__ __managed__ float c[m*n];

int main(void) {
	cublasStatus_t stat;					// CUBLAS functions status
	cublasHandle_t handle;					// CUBLAS context
	int i,j;								// i-row index,j-column index
	
	// define an mxk matrix a column by column
	int ind=11;								// a:
	for (j=0;j<k;j++) { 					// 11,17,23,29,35 
		for (i=0;i<m;i++) { 				// 12,18,24,30,36
			a[i +j*m]=(float) ind++;			// 13,19,25,31,37
		}									// 14,20,26,32,38
	}										// 15,21,27,33,39
											// 16,22,28,34,40
	
	// print a row by row
	std::cout << " a: " << std::endl;
	for (i=0;i<m;i++) {
		for (j=0;j<k;j++) {
			std::cout << a[i+j*m] << " " ; }
		std::cout << std::endl; 
	}
	// define a kxn matrix b column by column
	ind=11;									// b:
	for (j=0;j<n;j++) {						// 11,16,21,26
		for (i=0;i<k;i++) {					// 12,17,22,27
			b[i+j*k]=(float) ind++;			// 13,18,23,28 
		}									// 14,19,24,29
	}										// 15,20,25,30
	// print b row by row
	std::cout << " b: " << std::endl;
	for (i=0;i<k;i++) {
		for (j=0;j<n;j++) {
			std::cout << b[i+j*k] << " " ; } 
		std::cout << std::endl;
	}
	
	// define an mxn matrix c column by column
	ind=11;									// c:
	for (j=0;j<n;j++) {						// 11,17,23,29
		for (i=0;i<m;i++) {					// 12,18,24,30
			c[i+j*m]=(float)ind++;			// 13,19,25,31
		}									// 14,20,26,32
	}										// 15,21,27,33
											// 16,22,28,34
	// print c row by row
	std::cout << "c: " << std::endl;
	for (i=0;i<m;i++){
		for (j=0;j<n;j++) {
			std::cout << c[i +j*m] << " ";
		}
		std::cout << std::endl;
	}

	// Important to initialize CUBLAS context, creating the handle; otherwise Segmentation Fault
	stat = cublasCreate(&handle);			// initialize CUBLAS context
	

	float a1 = 1.0f;
	float bet=1.0f;
	// matrix-matrix multiplication: d_c = a1*a*b + bet*c
	// a - mxk matrix, b - kxn matrix, c - mxn matrix;
	// a1,bet- scalars

	stat=cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,&a1,a,m,b,k,&bet,c,m);
	
	/* I compiled this code on a 980Ti.								*/
	/* you can only do the following commented out code for Compute Capability 6.X and higher!  
		 
		* Please help me with obtaining a GTX 1080 Ti by donating at the PayPal link above so I can do so! -EY */
	/* These commands are needed for Compute Capability 5.2.  		*/

	float* host_c;							// host memory for c
	host_c=(float*)malloc(m*n*sizeof(float));
	stat=cublasGetMatrix(m,n,sizeof(*host_c),c,m,host_c,m);	// cp c -> host_c


	std::cout << " c after Sgemm : " << std::endl; 
	for (i=0;i<m;i++) {
		for (j=0;j<n;j++){
			std::cout << host_c[i+j*m] << " "; 
		}
		std::cout << std::endl;
	}

	cublasDestroy(handle); 	// destroy CUBLAS context
	return EXIT_SUCCESS;  

	
}
