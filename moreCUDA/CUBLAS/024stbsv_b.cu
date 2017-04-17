/**
 * @file   : 024stbsv_b.cu
 * @brief  : cublasStbsv - solve the triangular banded linear system  
 * uses CUDA Unified Memory (Management); bare minimum
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
// nvcc -std=c++11 -arch='sm_52' 024stbsv_b.cu -lcublas -o 024stbsv_b.exe
#include <iostream>
#include "cublas_v2.h"

constexpr const int n =6;			// number of rows and columns of a 
constexpr const int k =1;				// number of subdiagonals

__device__ __managed__ float A[n*n];  	// nxn matrix A on CUDA Unified (managed) memory 
__device__ __managed__ float b[n];	// n-vector b on CUDA Unified (managed) memory 
__device__ __managed__ float x[n];  // n-vector x on CUDA Unified (managed) memory

int main(void) {
	cudaError_t cudaStat;					// cudaMalloc status
	cublasStatus_t stat;					// CUBLAS functions status
	cublasHandle_t handle;					// CUBLAS context
	int i,j;							// lower triangle of a:
	
	// main diagonal and subdiagonals of A in rows:
	int ind=11;
	// main diagonal: 11, 12,13,14,15,16 in row 0
	for (i=0; i<n; i++) {
		A[i*n]=(float)ind++;  
	}
	
	// first subdiagonal: 17, 18, 19, 20, 21 in row 1
	for (i=0;i<n-1;i++) {
		A[i*n+1]=(float)ind++; 
	}
		
	for (i=0; i<n;i++) {
		b[i]=1.0f; 
	}					// b={1,1,1,1,1,1}^T

	for (j=0;j<n;j++){
		for (i=0;i<n;i++){
			std::cout << A[i + n*j] << " "; 
	} 
		std::cout << std::endl; }

	for (i=0;i<n; i++) {
		std::cout << b[i] << " "; } std::cout << std::endl;

		
	stat = cublasCreate(&handle);		// initialize CUBLAS context

	/* print out */
	for (j=0;j<n;j++){
		for (i=0;i<n;i++){
			std::cout << A[i + n*j] << " "; 
	} 
		std::cout << std::endl; }
	/* END of print out */


	// solve a triangular banded linear system: Ax=b;
	// the solution x overwrite the right hand side (RHS) b;
	// A - nxn banded lower triangular matrix; b - n-vector

	stat=cublasStbsv(handle,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_N,
							CUBLAS_DIAG_NON_UNIT, n,k, A,n, b,1);

	// for some reason, if cublasGetVector isn't included, Bus error (core dumped) isn't obtained
	stat = cublasGetVector(n,sizeof(float),b,1,x,1);

	// print the solution
	std::cout << "solution : " << std::endl; 	// print x after Stbsv
	for(j=0;j<n;j++) 
	{
		std::cout << b[j] << std::endl; }
	for(j=0;j<n;j++) 
	{
		std::cout << x[j] << std::endl; }



	cublasDestroy(handle);	// destroy CUBLAS context	
	
	cudaDeviceReset();
	return EXIT_SUCCESS;

}
