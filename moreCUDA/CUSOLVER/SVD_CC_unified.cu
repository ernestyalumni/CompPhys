/**
 * @file   : SVD_CC_unified.cu
 * @brief  : Simple Examples of calculating singular value decomposition (SVD) for complex numbers (CC)  
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
 * nvcc -std=c++11 -arch='sm_52' -lcudart -lcublas -lcusolver SVD_CC_unified.cu -o SVD_CC_unified.exe
 * */

#include <iostream> 	// std::cout
#include <iomanip> 		// std::setprecision 

// needed for SVD
#include <cuda_runtime.h>  // cudaError_t
#include <cusolverDn.h> // Dn = dense (matrices)

#include <cuComplex.h> // cuComplex, cuDoubleComplex

constexpr const int d = 2;  
constexpr const int L = 4; 

// all needed for SVD
__device__ __managed__ cuDoubleComplex Psi[ 1 << L];  // 1 << L = 2**L = 16 
__device__ __managed__ cuDoubleComplex U[ 1 << L]; // d**(L-1) x d unitary matrix
__device__ __managed__ cuDoubleComplex VT[ d*d ];  // d-by-d unitary matrix
__device__ __managed__ double S[ d ]; // singular value  
__device__ __managed__ int *devInfo = nullptr; 
__device__ __managed__ double *d_rwork = NULL; 


// boilerplate, initialization; matrix assumed to be column-major ordering 
void create_random_C_matrix(const int M, const int N, cuDoubleComplex * A) 
{
	for (int i=0; i < M; i++) { 
		for (int j=0; j<N; j++) { 
			A[i+j*M].x = cos((double) rand()/(RAND_MAX)*2.*acos(-1.));  // [0,1)*2*pi
			A[i+j*M].y = sin((double) rand()/(RAND_MAX)*2.*acos(-1.));	// [0,1)*2*pi
		}
	}

}
// boilerplate for printout
// usually the value of lda is m
void printMatrix(int m, int n, const cuDoubleComplex *A, int lda, const char* name) 
{
	std::cout << name << std::endl;
	for (int row =0; row <m; row++) {
		for (int col =0 ; col <n ; col++) {
			double Aregx = A[row + col*lda].x; 
			double Aregy = A[row + col*lda].y; 

			std::cout << std::setprecision(9) << Aregx << "+i"<< Aregy << " " ; 
		}
		std::cout << std::endl;
	}
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

/* ********************************************************************/
/* ******** MAIN 
 * ********************************************************************/

int main(int argc, char* argv[]) {
	// sanity check:
	std::cout << " L : " << L << " 1 << L : " << (1<<L) << std::endl << std::endl; 

	cusolverDnHandle_t cusolverH = NULL;
	cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;  
	
	// --- cuSOLVER input/output parameters/arrays
	// working space, <type> array of size lwork
	cuDoubleComplex *d_work = NULL;
	// size of working array work
	int lwork = 0;

// step 1: create cusolverDn/cublas handle, CUDA solver initialization
	cusolver_status = cusolverDnCreate(&cusolverH);

	// --- Setting the boilerplate, initialization values
	create_random_C_matrix( 1<< (L-1), d, Psi); 
	// sanity check
	printMatrix( 1<<(L-1), d, Psi, 1<<(L-1), "Psi");

// step 2: query working space of SVD	
	cusolver_status = cusolverDnZgesvd_bufferSize( cusolverH, 1<<(L-1), d, &lwork);  // lwork is 336 in this case
	cudaMalloc((void**)&d_work, sizeof(cuDoubleComplex)*lwork);

	cusolver_status = cusolverDnZgesvd(cusolverH,'A','A', 1<<(L-1),d, Psi, 1<<(L-1), 
		S,U, 
		1<<(L-1), // ldu
		VT, 
		d, // ldvt
//		d_work,lwork, d_rwork, devInfo);
		d_work,lwork, NULL, devInfo);


	
	printMatrix( 1<<(L-1), d, U, 1<<(L-1), " U ");
	std::cout << " U values : " << U[0].x << " " << U[0].y << " " << U[1].x << " " << U[1].y << std::endl;

	printMatrix( d, d, VT, d, " VT ");
	print1darr<double>( d, S, 1, " S : ");




// free resources
	if (cusolverH) cusolverDnDestroy(cusolverH);
	if (d_work) cudaFree(d_work);
	
	cudaDeviceReset();
	return 0;


}
