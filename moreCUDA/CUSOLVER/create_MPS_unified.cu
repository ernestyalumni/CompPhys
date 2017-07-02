/**
 * @file   : create_MPS_unified.cu
 * @brief  : Simple Examples of calculating matrix product states (MPS) using singular value decomposition   
 * this implementation uses CUDA Unified Memory Management
 *
 * @author : Ernest Yeung	ernestyalumni@gmail.com
 * @date   : 20170701
 * @ref    :  cf.arXiv:1008.3477 [cond-mat.str-el]
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
 * nvcc -std=c++11 -arch='sm_52' -lcudart -lcublas -lcusolver SVD_vectors_unified.cu -o SVD_unified.exe
 * */

#include <iostream> 	// std::cout
#include <iomanip> 		// std::setprecision 

#include <math.h> // pow

// needed for SVD
#include <assert.h> // assert
#include <cuda_runtime.h>  // cudaError_t
#include <cublas_v2.h>
#include <cusolverDn.h> // Dn = dense (matrices)

#include <cuComplex.h> // cuComplex, cuDoubleComplex


constexpr const int d = 2;  // dim. of state vector space 
constexpr const int L = 4;  // number of sites  

__device__ __managed__ float2 Psi[ 1 << L];
__device__ __managed__ float2 U[ 1 << L]; // d**(L-1) x d unitary matrix
__device__ __managed__ float2 VT[ d*d ];  // d-by-d unitary matrix
__device__ __managed__ float S[ d ]; // singular value  

// after cuSOLVER gesvd, we need to prepare for cuBLAS routines  
__device__ __managed__ cuComplex S2[d]; // singular values but different type 
__device__ __managed__ float2 US[ 1<<L ]; // singular value  
__device__ __managed__ int *devInfo = nullptr; 
__device__ __managed__ float *d_rwork = NULL; 

//__device__ __managed__ 

// boilerplate, initialization; matrix assumed to be column-major ordering 
void create_random_C_matrix(const int M, const int N, float2 * A) 
{
	for (int i=0; i < M; i++) { 
		for (int j=0; j<N; j++) { 
			A[i+j*M].x = cosf((float) rand()/(RAND_MAX)*2.f*acosf(-1.f));  // [0,1)*2*pi
			A[i+j*M].y = sinf((float) rand()/(RAND_MAX)*2.f*acosf(-1.f));	// [0,1)*2*pi
		}
	}

}
// boilerplate for printout
void printMatrix(int m, int n, const float2 *A, int lda, const char* name) 
{
	std::cout << name << std::endl;
	for (int row =0; row <m; row++) {
		for (int col =0 ; col <n ; col++) {
			float Aregx = A[row + col*lda].x; 
			float Aregy = A[row + col*lda].y; 

			std::cout << std::setprecision(9) << Aregx << "+i"<< Aregy << " " ; 
		}
		std::cout << std::endl;
	}
}

// Parameter:
// int lda - the "stride" between successive values of the array; e.g. lda=1 (usually this is the case)
template <typename TT>
void print2darr(const int N, const TT *A, int lda, const char* name) 
{
	std::cout << name << std::endl;
	for (int row =0; row < N; row++) {
		TT Areg = A[row *lda]; 
		std::cout << Areg.x << " " << Areg.y << " " ; 
	}
	std::cout << std::endl;
}

template <typename TT>
void print1darr(const int N, const TT *A, int lda, const char* name) 
{
	std::cout << name << std::endl;
	for (int row =0; row < N; row++) {
		TT Areg = A[row *lda]; 
		std::cout << Areg << " "  ; 
	}
	std::cout << std::endl;
}

int main( int argc, char* argv[] ) {
	cudaError_t cudaStat = cudaSuccess;  // cudaSuccess=0, cf. http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#axzz4lEpqZl2L


	cusolverDnHandle_t cusolverH = NULL;
	cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;  

	cublasHandle_t cublasH = NULL;
	cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;

	// --- cuSOLVER input/output parameters/arrays
	// working space, <type> array of size lwork
	float2 *d_work = NULL;
	// size of working array work
	int lwork = 0;


	std::cout << " d**L : " << (1 << L) << std::endl; 

	create_random_C_matrix(1<<(L-1),d,Psi);
	printMatrix( 1<<(L-1), d, Psi, 1<<(L-1), "Psi");

// step 1: create cusolverDn/cublas handle, CUDA solver initialization
	cusolver_status = cusolverDnCreate(&cusolverH);
	assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

	cublas_status = cublasCreate(&cublasH);
	assert(CUBLAS_STATUS_SUCCESS == cublas_status);


// step 2: query working space of SVD	
	cusolver_status = cusolverDnCgesvd_bufferSize( cusolverH, 1<<(L-1), d, &lwork);
	assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);
	std::cout << " \n lwork = " << lwork << std::endl << std::endl;
	cudaMalloc((void**)&d_work, sizeof(float2)*lwork);


	cusolver_status = cusolverDnCgesvd(cusolverH,'A','A', 1<<(L-1),d, Psi, 1<<(L-1), 
		S,U, 
		1<<(L-1), // ldu
		VT, 
		d, // ldvt
		d_work,lwork, NULL, devInfo);
//	assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);
	
	// sanity check
	print1darr<float>(d,S,d," S, after doing SVD " );
	
	// Step 5: US=U*S and B=VT
		// US = U*S

	// change singular values, 1-dim. array, of floats into 1-dim. array of cuComplex
	cudaStat = cudaMemcpy2D( S2, 			// dst - Destination memory address  
							 2*sizeof(float), 	// dpitch - Pitch of destination memory  
							 S, 			// src - Source memory address  
							1*sizeof(S[0]), 	// spitch - Pitch of source memory; in this case realx[0] is a float
							sizeof(S[0]), 		// width of matrix transfer (columns in bytes); in this case S[0] is a float
							d,						// height of matrix transfer (rows)
							cudaMemcpyDeviceToDevice);  

	assert(cudaSuccess == cudaStat); 
	cudaDeviceSynchronize();

	printMatrix(1<<(L-1), d, U, 1<<(L-1), " U after SVD: ");
	print2darr<cuComplex>( d, S2, d, " S2 " );
	

	cublas_status = cublasCdgmm(cublasH,CUBLAS_SIDE_RIGHT, 1<<(L-1),d,
								U,1<<(L-1), 
								S2,1,
								US,1<<(L-1));
	assert(CUBLAS_STATUS_SUCCESS == cublas_status);

	cudaDeviceSynchronize();  
	
	printMatrix( 1<<(L-1), d, US, d, " US ");


//	printMatrix( d, d, VT, d, "VT");

// free resources
	if (cublasH) cublasDestroy(cublasH);
	if (cusolverH) cusolverDnDestroy(cusolverH);
	if (d_work) cudaFree(d_work);
	
	cudaDeviceReset();
	return 0;

	
}
