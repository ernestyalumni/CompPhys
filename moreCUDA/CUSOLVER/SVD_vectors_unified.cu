/**
 * @file   : SVD_vectors_unified.cu
 * @brief  : Simple example in C of singular value decomposition, but with singular vectors
 *
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
 * nvcc -std=c++11 -arch='sm_52' -lcudart -lcublas -lcusolver SVD_vectors_unified.cu -o SVD_vectors_unified.exe
 * */

#include <iostream> 	// std::cout
#include <iomanip> 		// std::setprecision 

#include <assert.h> // assert
#include <cuda_runtime.h>  // cudaError_t
#include <cublas_v2.h>
#include <cusolverDn.h> // Dn = dense (matrices)

constexpr const int m = 3;
constexpr const int n = 2;
constexpr const int lda = m;

__device__ __managed__ float A[lda*n] = { 1.0f, 4.0f, 2.0f, 2.0f, 5.0f, 1.0f };
__device__ __managed__ float U[lda*m]; // m-by-m unitary matrix
__device__ __managed__ float VT[lda*n]; // n-by-n unitary matrix
__device__ __managed__ float S[n]; 	// singular value
__device__ __managed__ int *devInfo = nullptr; 
__device__ __managed__ float W[lda*n]; 	// W = S*VT

__device__ __managed__ float *d_rwork = NULL; 

// Looks like this is for boilerplate and it looks like that 
/* 
 * lda = stride
 * it's in "column-major" order; cuSOLVER assumes for dense matrices COLUMN-major order
 * cf. http://docs.nvidia.com/cuda/cusolver/index.html#format-dense-matrix
 * */
void printMatrix(int m, int n, const float *A, int lda, const char* name) 
{
	std::cout << name << std::endl;
	for (int row =0; row <m; row++) {
		for (int col =0 ; col <n ; col++) {
			float Areg = A[row + col*lda]; 
			std::cout << Areg << " " ; 
		}
		std::cout << std::endl;
	}
}

int main(int argc, char* argv[]) {
	/** 
	 * @name cusolverDnCreate(cusolverDnHandle_t *handle);
	 * @brief This function initializes the cuSolverDN library and creates a handle on the cuSolverDN context. 
	 * 	It must be called before any other cuSolverDN API function is invoked.  It allocates hardware resources 
	 * 	necessary for accessing the GPU.  
	 * cuSolverDN: dense LAPACK, dense LAPACK functions, as opposed to SP, sparse, RF refactorization 
	 * */

	cusolverDnHandle_t cusolverH = NULL;
	
	cublasHandle_t cublasH = NULL;

	cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
	cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS; 
	
	/*		| 1 2 	| 
	 * 	A = | 4 5 	| 
	 * 		| 2 1 	| 
	 * */

	// working space, <type> array of size lwork
	float *d_work = NULL;

	// size of working array work
	int lwork = 0;
	 
	const float h_one = 1.f;
	const float h_minus_one = -1.f;  
	 
	float S_exact[n] = {7.065283497082729f, 1.040081297712078f};


	std::cout << " A = (matlab base-1) " << std::endl; 
	printMatrix(m, n, A, lda, "A");
	std::cout << " ===== " << std::endl; 

	cudaDeviceSynchronize();  

// step 1: create cusolverDn/cublas handle 
	cusolver_status = cusolverDnCreate(&cusolverH);
	assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
	
	cublas_status = cublasCreate(&cublasH);
	assert(CUBLAS_STATUS_SUCCESS == cublas_status);

// step 2: copy A and B to device is handled automatically by managed device

// step 3: query working space of SVD

	// The S and D data types are real valued single and double precision, respectively
	/** cusolverDnSgesvd_bufferSize 
	 * @brief Calculate size of work buffer used by cusolverDnDgesvd.
	 * @ref http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-gesvd
	 * 
	 * cusolverStatus_t
	 * cusolverDnSgesvd_bufferSize(
	 * 	cusolverDnHandle_t handle,
	 * 	int m,
	 * 	int n,
	 * 	int *lwork);
	 * */

	// The S and D data types are real valued single and double precision, respectively
	cusolver_status = cusolverDnSgesvd_bufferSize( cusolverH, m, n, &lwork);
	assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);
	std::cout << " \n lwork = " << lwork << std::endl << std::endl; 
	cudaMalloc((void**)&d_work , sizeof(float)*lwork);
	
	
	// step 4: compute SVD
	/**
	 * jobu, input
	 * @brief specifies options for computing all or part of the matrix U:= 'A': all m columns of U are returned in array 
	 * 	U:='S': the first min(m,n) columns of U (the left singular vectors) are returned in the array
	 * 	U:='O': the first min(m,n) columns of U (the left singular vectors) are overwritten on the array A; 
	 * 	 = 'N': no columns (no left singular vectors) are computed.  
	 * 
	 * */
	signed char jobu = 'A'; // all m columns of U 
	/**
	 * jobvt, input
	 * @brief specifies options for computing all or part of the matrix V**T:
	 * ='A': all N rows of V**T are returned in the array VT;
	 * ='S': the first min(m,n) rows of V**T (the right singular vectors) are returned in the array VT;
	 * ='O': the first min(m,n) rows of V**T (the first singular vectors) are overwritten on the array A;
	 * ='N': no rows of V**T (no right singular vectors) are computed.  
	 * */
	signed char jobvt = 'A'; // all n columns of VT

	/**
	 * cusolverDnSgesvd - computes the singular value decomposition (SVD) of 
	 * mxn matrix A
	 * and corresponding left and/or right singular vectors
	 * SVD written 
	 * A = U S V^H
	 * where 
	 * S = m x n matrix which is 0, except for its min(m,n) diagonal elements  
	 * U = m x m unitary matrix  
	 * V = n x n unitary matrix
	 * Diagonal elements of S are singular values of A; they are real and non-negative, and returned in descending order
	 * The first min(m,n) columns of U and V are left and right singular vectors of A 
	 * 
	 * @name API of gesvd (partial API)
	 * @brief ldu - input - leading dimension of 2-dim. array used to store matrix U
	 * ldvt - input - leading dim. of 2-dim. array used to store matrix Vt
	 * rwork (here, it is d_rwork) - device - input - real array of dim. min(m,n)-1.  It contains the 
	 * 	unconverged superdiagonal elements of an upper bidiagonal matrix if devInfo > 0 
	 * devInfo - device - output - if devInfo = 0, the operation is successful, 
	 * 	if devInfo = -i, the i-th parameter is wrong. 
	 * 	if devInfo > 0, devInfo indicates how many superdiagonals of an intermediate bidiagonal form did not converge to 0
	 * */
	// The S and D data types are real valued single and double precision, respectively
	cusolver_status = cusolverDnSgesvd( cusolverH, jobu, jobvt, m, n, A, lda, S, U, 
		lda, // ldu 
		VT,
		lda, // ldvt
		d_work, lwork, d_rwork, devInfo);

	
	cudaDeviceSynchronize();

//	assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);  Assertion failed

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
	printMatrix(n,n,VT,lda,"VT");
	std::cout << " ====== " << std::endl; 
	

// step 5: measure error of singular value
	float ds_sup = 0.f;
	for (int j =0; j < n; j++) { 
		float err = fabs( S[j] - S_exact[j]);
		ds_sup = (ds_sup  > err) ? ds_sup : err;
	}
	std::cout << " |S-S_exact| = " << std::setprecision(9) << ds_sup << std::endl; 
	
// step 6: |A- U*S*VT |
		// W = S*VT

	/**
	 * cublas<t>dgmm()
	 * cublasSdgmm single float
	 * cublasDdgmm double float
	 * cublasCdgmm complex number
	 * cublasZdgmm
	 * @brief matrix-matrix multiplication
	 * @ref 2.8.2. cublas<t>dgmm() http://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-dgmm
	 * performs
	 * C = A x diag(X) if mode == CUBLAS_SIDE_RIGHT
	 * C = diag(X) x A if mode == CUBLAS_SIDE_LEFT
	 * 
	 * where A and C are matrices stored in column-major format with dims. m x n. 
	 * X is vector of size n if mode == CUBLAS_SIDE_RIGHT and 
	 * of size m if mode == CUBLAS_SIDE_LEFT.   
	 * X is gathered from 1-dim. array x with stride incc 
	 * 
	 * cublasStatust cublasSdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m, int n,
	 * const float * A, int lda, 
	 * const float *x, int incx, 
	 * float *C, int ldc)
	 * */

	cublas_status = cublasSdgmm( cublasH, CUBLAS_SIDE_LEFT, n,n,VT,lda,S,1,W,lda);
	assert(CUBLAS_STATUS_SUCCESS == cublas_status);

	/* sanity check
	cudaDeviceSynchronize();
	for (int idx=0;idx<4;idx++) { std::cout << W[idx] << " "; }
	cudaDeviceSynchronize();
	printMatrix(m, n, A, lda, "A");
	cudaDeviceSynchronize();
	*/
	
	/* EY : 20170628 I found that these steps are needed because A changed due to steps above.  */
	float host_A[lda*n] = { 1.0, 4.0, 2.0, 2.0, 5.0, 1.0 };
	cudaMemcpy(A, host_A,sizeof(float)*lda*n,cudaMemcpyHostToDevice);
	
	
	// A := -U*W + A
	/**
	 * @name cublass<t>gemm()
	 * @ref http://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-gemm
	 * cublasStatus_t cublasSgemm(cublasHandle_t handle, 
	 * 								cublasOperation_t transa, cublasOperation_t transb,
	 * 								int m, int n, int k,
	 * 								const float *alpha,
	 * 								const float *A, int lda,
	 * 								const float *B, int ldb,
	 * 								const float *beta,
	 * 								float *C, int ldc)
	 * @brief This function performs the matrix-matrix multiplication
	 * C = \alpha op(A)op(B) + \beta C
	 * 
	 * */
	cublas_status = cublasSgemm_v2(cublasH, 
									CUBLAS_OP_N, // U
									CUBLAS_OP_N, // W
									m, // number of rows of A
									n, // number of columns of A
									n, // number of columns of U
									&h_minus_one, /* host pointer */
									U, // U
									lda, 
									W, // W
									lda, 
									&h_one, /* host pointer */
									A, lda);
	assert(CUBLAS_STATUS_SUCCESS == cublas_status);
	
	float dR_fro = 0.0f;
	
	/**
	 * @name cublas<t>nrm2()
	 * @ref http://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-nrm2
	 * @brief This function computes the Euclidean norm of the vector x.  
	 * The code uses a multiphase model of accumulation to avoid intermediate underflow and overflow (EY : 20170628 what's under and over flow?)
	 * with the result being equivalent to sqrt( \sum_{i=1}^n (x[j] x x[j]) } 
	 * where j = 1+(i-1)*incx in exact arithmetic.  
	 * Notice that the last equation reflects 1-based indexing used for compatibility with Fortran.  
	 *   
	 * cublasStatus_t cublasSnrm2(cublasHandle_t handle, int n,
	 * 									const float *x, int incx, float *result)
	 * result - host or device - output - the result norm, which is 0.0 if n, incx <=0
	 * 
	 * */
			
	cublas_status = cublasSnrm2_v2(cublasH, lda*n, A, 1, &dR_fro);
	assert(CUBLAS_STATUS_SUCCESS == cublas_status);
	
	std::cout << "|A - U*S*VT| = " << std::setprecision(9) << dR_fro << std::endl;		
	
		
// free resources
	if (d_work) { cudaFree(d_work); }
		
	if (cublasH) cublasDestroy(cublasH);

	/**
	 * @name cusolverDnDestroy()
	 * cusolverStatus_t
	 * cusolverDnDestroy(cusolverDnHandle_t handle);
	 * 
	 * @brief This function release CPU-side resources used by the cuSolverDN library
	 * cuSolverDN dense LAPACK Function, library, as opposed to Sp sparse, RF, refactorization 
	 * @ref http://docs.nvidia.com/cuda/cusolver/index.html#cuSolverDNdestroy  
	 * 
	 * */
	if (cusolverH) cusolverDnDestroy(cusolverH);	
		
	cudaDeviceReset();
	return 0;
}
