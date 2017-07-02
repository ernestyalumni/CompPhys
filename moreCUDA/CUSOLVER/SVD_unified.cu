/**
 * @file   : SVD_unified.cu
 * @brief  : Example of singular value decomposition of a real matrix  
 * I compare this implementation, that uses CUDA Unified Memory Management, against that by OrangeOwl
 *
 * 				compute A = U*S*VT
 * @author : Ernest Yeung	ernestyalumni@gmail.com
 * @date   : 20170628
 * @ref    :  cf. https://github.com/OrangeOwlSolutions/Linear-Algebra/wiki/SVD-of-a-real-matrix
 * cf. https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-241j-dynamic-systems-and-control-spring-2011/readings/MIT6_241JS11_chap04.pdf
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
 * EY : 20170628 This also worked for me
 * nvcc -std=c++11 -arch='sm_52' -lcudart -lcublas -lcusolver SVD_vectors_unified.cu -o SVD_vectors_unified.exe
 * */
 
#include <iostream> 	// std::cout
#include <iomanip> 		// std::setprecision 

#include <stdio.h> // printf

#include <math.h> 	// sqrt

#include <assert.h> // assert
#include <cuda_runtime.h>  // cudaError_t
#include <cublas_v2.h>
#include <cusolverDn.h> // Dn = dense (matrices)

#include <array> // std::array

constexpr const int m = 7;
constexpr const int n = 5;
constexpr const int lda = m;

__device__ __managed__ double A[m*n] ;
__device__ __managed__ double U[lda*m]; // m-by-m unitary matrix
__device__ __managed__ double VT[n*n]; // n-by-n unitary matrix
__device__ __managed__ double S[n]; 	// singular value
__device__ __managed__ int *devInfo = nullptr; 
__device__ __managed__ double *d_rwork = NULL; 

// For more examples  
__device__ __managed__ double A2[2*2]; 
__device__ __managed__ double U2[2*2]; 
__device__ __managed__ double VT2[2*2]; 
__device__ __managed__ double S2[2]; 
__device__ __managed__ int *devInfo2 = nullptr; 


// for checking the error
__device__ __managed__ double W[lda*n]; 	// W = S*VT


// Looks like this is for boilerplate and it looks like that 
/* 
 * lda = stride
 * it's in "column-major" order; cuSOLVER assumes for dense matrices COLUMN-major order
 * cf. http://docs.nvidia.com/cuda/cusolver/index.html#format-dense-matrix
 * */
void printMatrix(int m, int n, const double *A, int lda, const char* name) 
{
	std::cout << name << std::endl;
	for (int row =0; row <m; row++) {
		for (int col =0 ; col <n ; col++) {
			double Areg = A[row + col*lda]; 
			std::cout << std::setprecision(9) << Areg << " " ; 
		}
		std::cout << std::endl;
	}
}

int main(int argc, char* argv[]) {

	cusolverDnHandle_t cusolverH = NULL;
	cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
	
	cublasHandle_t cublasH = NULL;
	cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
	
	// --- cuSOLVER input/output parameters/arrays
	// working space, <type> array of size lwork
	double *d_work = NULL;
	// size of working array work
	int lwork = 0;
	
// step 1: create cusolverDn/cublas handle, CUDA solver initialization
	cusolver_status = cusolverDnCreate(&cusolverH);
	assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
	
	cublas_status = cublasCreate(&cublasH);
	assert(CUBLAS_STATUS_SUCCESS == cublas_status);
	
	// --- Setting the boilerplate, initialization values
	for (int i=0; i<m; i++) {
		for (int j=0; j < n; j++) {
			A[i+j*m] = (i*i + j) * sqrt((double)(i+j));
		}
	}

	/* sanity check
	cudaDeviceSynchronize(); 
	printMatrix(m,n,A,lda,"A");
	*/

// step 2: query working space of SVD	
//	cusolver_status = cusolverDnSgesvd_bufferSize( cusolverH, m, n, &lwork);
	cusolver_status = cusolverDnDgesvd_bufferSize( cusolverH, m, n, &lwork);
	
	assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);
	std::cout << " \n lwork = " << lwork << std::endl << std::endl;
	cudaMalloc((void**)&d_work, sizeof(double)*lwork);

// step 4: compute SVD
	signed char jobu = 'A'; // all m columns of U
	signed char jobvt = 'A';  // all n columns of VT

	// --- CUDA SVD execution
	cusolver_status = cusolverDnDgesvd(cusolverH,jobu,jobvt, m,n, A, lda, 
		S,U, 
		lda, // ldu
		VT, 
		n, // ldvt
		d_work,lwork, d_rwork, devInfo);

	cudaDeviceSynchronize();
	
	// --- results from SVD
	
	std::cout << " cusolver_status after SVD : " << cusolver_status << std::endl;
	
	std::cout << " after gesvd: info_gpu or devInfo = " << devInfo << std::endl ; 
	assert(0 == devInfo);
	std::cout << " ====== " << std::endl; 

	std::cout << " S = (matlab base-1) : " << std::endl; 
	printMatrix(n,1,S,lda,"S");
	std::cout << " ====== " << std::endl; 
	
	std::cout << " U = (matlab base-1) : " << std::endl; 
	printMatrix(m,m,U,lda,"U");
	std::cout << " ====== " << std::endl; 
	
	std::cout << " VT = (matlab base-1) : " << std::endl; 
	printMatrix(n,n,VT,n,"VT");
	std::cout << " ====== " << std::endl; 
	
/* ********************** ERROR measurement ************************* */
	// Step 5: |A-U*S*VT|
		// W = S*VT
	cublas_status = cublasDdgmm(cublasH,CUBLAS_SIDE_LEFT,n,n,VT,n,S,1,W,lda);
	assert(CUBLAS_STATUS_SUCCESS == cublas_status);

	// reset A
	std::array<double,m*n> host_A_vec;

	for (int i=0; i<m; i++) {
		for (int j=0; j < n; j++) {
			host_A_vec[i+j*m] = (i*i + j) * sqrt((double)(i+j));
		}
	}
	cudaMemcpy(A,host_A_vec.data(), sizeof(double)*m*n,cudaMemcpyHostToDevice);
	
	

	// A := -U*W + A

	double h_minus_1 = -1.;
	double h_1       = 1.;
	cublas_status= cublasDgemm_v2(cublasH, CUBLAS_OP_N,CUBLAS_OP_N, m,n,
									n, // number of columns of U 
									&h_minus_1, /* host pointer */
									U, lda, W, lda, &h_1, A, lda);
	assert(CUBLAS_STATUS_SUCCESS == cublas_status);

	double dR_fro = 0.0;
	
	cublas_status=cublasDnrm2_v2(cublasH,m*n,A, 1, &dR_fro);
	assert(CUBLAS_STATUS_SUCCESS == cublas_status);
	
	std::cout << "|A - U*S*VT| = " << std::setprecision(9) << dR_fro << std::endl;



	/* ****************************************************************/
	/* ***** More examples
	 * cf. ref:
	/* ****************************************************************/
	// --- Setting the boilerplate, initialization values
	std::cout << "\n Examples from elsewhere : " << std::endl; 
	A2[0] = 100. ;
	A2[1] = 100.2 ;
	A2[2] = 100.;
	A2[3] = 100.;
	printMatrix(2,2,A2,2,"A2");
	
	// --- cuSOLVER input/output parameters/arrays
	// working space, <type> array of size lwork
	double *d_work2 = NULL;
	// size of working array work
	int lwork2 = 0;

	
	cusolverDnHandle_t cusolverH2 = NULL;
// step 1: create cusolverDn/cublas handle, CUDA solver initialization
	cusolver_status = cusolverDnCreate(&cusolverH2);


	// step 2: query working space of SVD	
	cusolver_status = cusolverDnDgesvd_bufferSize( cusolverH2, 2, 2, &lwork2);
	cudaMalloc((void**)&d_work2, sizeof(double)*lwork2);


	cudaDeviceSynchronize();
	// step 4: compute SVD
	cusolver_status = cusolverDnDgesvd(cusolverH2,'A','A', 2,2, A2, 2, 
		S2,U2, 
		2, // ldu
		VT2, 
		2, // ldvt
		d_work2,lwork2, NULL, devInfo2);

	cudaDeviceSynchronize();


	std::cout << " S2 = (matlab base-1) : " << std::endl; 
	printMatrix(2,1,S2,2,"S2");
//	std::cout << S2[0] << " " << S2[1] << std::endl; 
	printf(" %f %f \n", S2[0], S2[1] ); 

	std::cout << " ====== " << std::endl; 
	
	
	std::cout << " U2 = (matlab base-1) : " << std::endl; 
	printMatrix(2,2,U2,2,"U2");
	std::cout << " ====== " << std::endl; 
	
	std::cout << " VT2 = (matlab base-1) : " << std::endl; 
	printMatrix(2,2,VT2,2,"VT2");
	std::cout << " ====== " << std::endl; 





// free resources
	if (cublasH) cublasDestroy(cublasH);
	if (cusolverH) cusolverDnDestroy(cusolverH);
	if (cusolverH2) cusolverDnDestroy(cusolverH2);
	if (d_work) cudaFree(d_work);
	if (d_work2) cudaFree(d_work2);
	
	cudaDeviceReset();
	return 0;
	
}
	
