/**
 * @file   : SVD_singular_values_only_unified.cu
 * @brief  : Example of comparing singular value decomposition of a real matrix  
 * I compare this implementation, that uses CUDA Unified Memory Management, against that by OrangeOwl
 *
 * 				compute A = U*S*VT
 * @author : Ernest Yeung	ernestyalumni@gmail.com
 * @date   : 20170628
 * @ref    :  cf. https://github.com/OrangeOwlSolutions/Linear-Algebra/wiki/SVD-of-a-real-matrix
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

#include <math.h> 	// sqrt

#include <assert.h> // assert
#include <cuda_runtime.h>  // cudaError_t
#include <cublas_v2.h>
#include <cusolverDn.h> // Dn = dense (matrices)

#include "gputimer.h" // GpuTimer

constexpr const int m = 1000;
constexpr const int n = 1000;
constexpr const int lda = m;

__device__ __managed__ double A[m*n] ;
__device__ __managed__ double U[lda*m]; // m-by-m unitary matrix
__device__ __managed__ double VT[n*n]; // n-by-n unitary matrix
__device__ __managed__ double S[n]; 	// singular value
__device__ __managed__ int *devInfo = nullptr; 
__device__ __managed__ double *d_rwork = NULL; 


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
			float Areg = A[row + col*lda]; 
			std::cout << Areg << " " ; 
		}
		std::cout << std::endl;
	}
}

int main(int argc, char* argv[]) {

	cusolverDnHandle_t cusolverH = NULL;
	cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;  
	
	// --- cuSOLVER input/output parameters/arrays
	// working space, <type> array of size lwork
	double *d_work = NULL;
	// size of working array work
	int lwork = 0;
	
// step 1: create cusolverDn/cublas handle, CUDA solver initialization
	cusolver_status = cusolverDnCreate(&cusolverH);
	assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

	// --- Setting the boilerplate, initialization values
	for (int i=0; i<m; i++) {
		for (int j=0; j < n; j++) {
			A[i+j*m] = (i*i + j) * sqrt((double)(i+j));
		}
	}

// step 2: query working space of SVD	
	cusolver_status = cusolverDnSgesvd_bufferSize( cusolverH, m, n, &lwork);
	assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);
	std::cout << " \n lwork = " << lwork << std::endl << std::endl;
	cudaMalloc((void**)&d_work, sizeof(double)*lwork);

// step 4: compute SVD
	GpuTimer timer;

	// --- CUDA SVD execution - Singular values only
		/**
	 * jobu, input
	 * @brief specifies options for computing all or part of the matrix U:= 'A': all m columns of U are returned in array 
	 * 	U:='S': the first min(m,n) columns of U (the left singular vectors) are returned in the array
	 * 	U:='O': the first min(m,n) columns of U (the left singular vectors) are overwritten on the array A; 
	 * 	 = 'N': no columns (no left singular vectors) are computed.  
	 * 
	 * */
	timer.Start();
	cusolver_status = cusolverDnDgesvd(cusolverH,'N','N', m,n, A, lda, 
		S,U, 
		lda, // ldu
		VT, 
		n, // ldvt
		d_work,lwork, d_rwork, devInfo);
	timer.Stop();
	std::cout << " Calculation of singular values only : " << std::setprecision(7) << timer.Elapsed() << " ms " << std::endl;
	
	
	// --- CUDA SVD execution - Full SVD	
	
	timer.Start();

	cusolver_status = cusolverDnDgesvd(cusolverH,'A','A', m,n, A, lda, 
		S,U, 
		lda, // ldu
		VT, 
		n, // ldvt
		d_work,lwork, d_rwork, devInfo);
	timer.Stop();
	
	std::cout << " Calculation of the full SVD calculation : " << std::setprecision(7) << timer.Elapsed() << " ms " << std::endl;
	
	
// free resources
	if (cusolverH) cusolverDnDestroy(cusolverH);
	if (d_work) cudaFree(d_work);
	
	cudaDeviceReset();
	return 0;

}
