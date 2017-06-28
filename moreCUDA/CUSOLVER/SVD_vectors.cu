/**
 * @file   : SVD_vectors.cu
 * @brief  : Simple example in C of singular value decomposition, but with singular vectors
 * 				compute A = U*S*VT
 * @author : Ernest Yeung	ernestyalumni@gmail.com
 * @date   : 20170627
 * @ref    :  cf. http://docs.nvidia.com/cuda/cusolver/index.html#svd_examples
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

/*
 * How to compile (assume cuda is installed at /usr/local/cuda/)
 * 	 nvcc -c -I/usr/local/cuda/include svd_example.cpp
 * 	 g++ -fopenmp -o a.out svd_example.o -L/usr/local/cuda/lib64 -lcudart -lcublas -lcusolver
 * 
 * EY : 20170627 This also worked for me
 * nvcc -lcudart -lcublas -lcusolver SVD_vectors.cu -o SVD_vectors.exe
 * */
 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <cuda_runtime.h>  // cudaError_t
#include <cublas_v2.h>
#include <cusolverDn.h> // Dn = dense (matrices)

// Looks like this is for boilerplate and it looks like that 
/* 
 * lda = stride
 * it's in "column-major" order; cuSOLVER assumes for dense matrices COLUMN-major order
 * cf. http://docs.nvidia.com/cuda/cusolver/index.html#format-dense-matrix
 * */
void printMatrix(int m, int n, const double *A, int lda, const char* name) 
{
	for (int row =0; row <m; row++) {
		for (int col =0 ; col <n ; col++) {
			double Areg= A[row + col*lda]; 
			printf("%s(%d,%d) = %f\n", name, row+1,col+1, Areg); 
		}
	}
}

int main(int argc, char* argv[]) {
	cusolverDnHandle_t cusolverH = NULL;
	
	cublasHandle_t cublasH = NULL;

	cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
	
	cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
	
	cudaError_t cudaStat1 = cudaSuccess;  // cudaSuccess=0, cf. http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#axzz4lEpqZl2L
	cudaError_t cudaStat2 = cudaSuccess;
	cudaError_t cudaStat3 = cudaSuccess;
	cudaError_t cudaStat4 = cudaSuccess;
	cudaError_t cudaStat5 = cudaSuccess;
	cudaError_t cudaStat6 = cudaSuccess;
	
	const int m = 3;
	const int n = 2;
	const int lda = m;
	
	/*		| 1 2 	| 
	 * 	A = | 4 5 	| 
	 * 		| 2 1 	| 
	 * */
	 
	double A[lda*n] = { 1.0, 4.0, 2.0, 2.0, 5.0, 1.0 };
	double U[lda*m]; // m-by-m unitary matrix
	double VT[lda*n]; // n-by-n unitary matrix
	double S[n]; 	// singular value
	double S_exact[n] = {7.065283497082729, 1.040081297712078};
		
	double *d_A = NULL;
	double *d_S = NULL;
	double *d_U = NULL; 
	double *d_VT = NULL; 
	int *devInfo = NULL;
	double *d_work = NULL;
	double *d_rwork = NULL;
	double *d_W = NULL; // W = S*VT
	
	int lwork = 0;
	int info_gpu = 0;
	const double h_one = 1;
	const double h_minus_one = -1; 
	
	printf("A = (matlab base-1)\n");
	printMatrix(m, n, A, lda, "A");
	printf("=====\n");
	
// step 1: create cusolverDn/cublas handle 
	cusolver_status = cusolverDnCreate(&cusolverH);
	assert (CUSOLVER_STATUS_SUCCESS == cusolver_status);
	
	cublas_status = cublasCreate(&cublasH);
	assert(CUBLAS_STATUS_SUCCESS == cublas_status);
	
// step 2: copy A and B to device
	cudaStat1 = cudaMalloc((void**)&d_A , sizeof(double)*lda*n);
	cudaStat2 = cudaMalloc((void**)&d_S , sizeof(double)*n);
	cudaStat3 = cudaMalloc((void**)&d_U , sizeof(double)*lda*m);
	cudaStat4 = cudaMalloc((void**)&d_VT , sizeof(double)*lda*n);
	cudaStat5 = cudaMalloc((void**)&devInfo, sizeof(int));
	cudaStat6 = cudaMalloc((void**)&d_W, sizeof(double)*lda*n);
	assert(cudaSuccess == cudaStat1);
	assert(cudaSuccess == cudaStat2);
	assert(cudaSuccess == cudaStat3);
	assert(cudaSuccess == cudaStat4);
	assert(cudaSuccess == cudaStat5);
	assert(cudaSuccess == cudaStat6);  
		
	cudaStat1 = cudaMemcpy(d_A, A, sizeof(double)*lda*n, cudaMemcpyHostToDevice);
	assert(cudaSuccess == cudaStat1);
	
// step 3: query working space of SVD 
	cusolver_status = cusolverDnDgesvd_bufferSize(
		cusolverH,
		m,
		n,
		&lwork );
	assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);
	
	cudaStat1 = cudaMalloc((void**)&d_work , sizeof(double)*lwork);
	assert(cudaSuccess == cudaStat1);
	
// step 4: compute SVD 
	signed char jobu = 'A'; // all m columns of U
	signed char jobvt = 'A'; // all n columns of VT
	cusolver_status = cusolverDnDgesvd(
		cusolverH,
		jobu,
		jobvt,
		m,
		n,
		d_A,
		lda,
		d_S,
		d_U,
		lda, 	// ldu
		d_VT,
		lda, 	// ldvt,
		d_work,
		lwork, 
		d_rwork,
		devInfo);
	cudaStat1 = cudaDeviceSynchronize();
	assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
	assert(cudaSuccess == cudaStat1);
	cudaStat1 = cudaMemcpy(U , d_U , sizeof(double)*lda*m, cudaMemcpyDeviceToHost);
	cudaStat2 = cudaMemcpy(VT , d_VT , sizeof(double)*lda*n, cudaMemcpyDeviceToHost);
	cudaStat3 = cudaMemcpy(S , d_S , sizeof(double)*n, cudaMemcpyDeviceToHost);
	cudaStat4 = cudaMemcpy(&info_gpu , devInfo , sizeof(int), cudaMemcpyDeviceToHost);
	assert(cudaSuccess == cudaStat1);
	assert(cudaSuccess == cudaStat2);
	assert(cudaSuccess == cudaStat3);
	assert(cudaSuccess == cudaStat4);
	
	printf("after gesvd: info_gpu = %d\n", info_gpu);
	assert(0 == info_gpu);
	printf("=====\n");
	
	printf("S = (matlab base-1)\n");
	printMatrix(n, 1, S, lda, "S");
	printf("=====\n");
	
	printf("U = (matlab base-1)\n");
	printMatrix(m, m, U, lda, "U");
	printf("=====\n");
	
	printf("VT = (matlab base-1)\n");
	printMatrix(n, n, VT, lda, "VT");
	printf("=====\n");
	
// step 5: measure error of singular value
	double ds_sup = 0;
	for (int j = 0; j < n; j++) {
		double err = fabs( S[j] - S_exact[j] ); 
		ds_sup = (ds_sup > err) ? ds_sup : err; 
	}
	printf("|S - S_exact| = %E \n", ds_sup);
	
// step 6: |A - U*S*VT|
	// W = S*VT
	cublas_status = cublasDdgmm(
		cublasH,
		CUBLAS_SIDE_LEFT,
		n,
		n,
		d_VT,
		lda,
		d_S,
		1,
		d_W,
		lda);
	assert(CUBLAS_STATUS_SUCCESS == cublas_status);
	
	// A := - U*W + A
	cudaStat1 = cudaMemcpy(d_A, A, sizeof(double)*lda*n, cudaMemcpyHostToDevice);
	assert(cudaSuccess == cudaStat1);
	cublas_status = cublasDgemm_v2(
		cublasH,
		CUBLAS_OP_N, // U
		CUBLAS_OP_N, // W
		m, // number of rows of A
		n, // number of columns of A
		n, // number of columns of U
		&h_minus_one, /* host pointer */ 
		d_U, // U
		lda,
		d_W, // W
		lda, 
		&h_one, /* hostpointer */
		d_A,
		lda);
	assert(CUBLAS_STATUS_SUCCESS == cublas_status);
	
	double dR_fro = 0.0;
	cublas_status = cublasDnrm2_v2(
		cublasH, lda*n, d_A, 1, &dR_fro);
	assert(CUBLAS_STATUS_SUCCESS == cublas_status);
	
	printf("|A - U*S*VT| = %E \n", dR_fro);
	
// free resources
	if (d_A		) cudaFree(d_A);
	if (d_S		) cudaFree(d_S);
	if 	(d_U 	) cudaFree(d_U);
	if	(d_VT	) cudaFree(d_VT);
	if (devInfo ) cudaFree(devInfo);
	if (d_work	) cudaFree(d_work);
	if (d_rwork	) cudaFree(d_rwork);
	if (d_W 	) cudaFree(d_W);
	
	if (cublasH	) cublasDestroy(cublasH);
	if (cusolverH) cusolverDnDestroy(cusolverH);
	
	


	cudaDeviceReset();
	return 0;
}
