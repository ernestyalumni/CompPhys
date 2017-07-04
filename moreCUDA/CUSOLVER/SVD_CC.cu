/**
 * @file   : SVD_CC.cu
 * @brief  : Simple example in C of singular value decomposition, but with singular vectors
 * 				compute A = U*S*VT
 * @author : Ernest Yeung	ernestyalumni@gmail.com
 * @date   : 20170703
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
#include <iostream> 	// std::cout
#include <iomanip> 		// std::setprecision 

#include <assert.h>
#include <cuda_runtime.h>  // cudaError_t
#include <cusolverDn.h> // Dn = dense (matrices)

#include <cuComplex.h> // cuComplex, cuDoubleComplex

void printMatrix(int m, int n, const cuDoubleComplex *A, int lda, const char* name) 
{
	std::cout << name << std::endl;
	for (int row =0; row <m; row++) {
		for (int col =0 ; col <n ; col++) {
			cuDoubleComplex Areg = A[row + col*lda]; 
			std::cout << std::setprecision(9) << Areg.x << "+i" << Areg.y << " " ; 
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

void create_linearval_C_matrix(const int M, const int N, cuDoubleComplex *A) {
	double ind_CC = 0.1; // value to scale the imaginary parts values by
	for (int i=0; i < M; i++) { 
		for (int j=0; j<N; j++) { 
			A[i+j*M].x = ((double) (i+1+ M*j));
			A[i+j*M].y = ind_CC * ( (double) i+1 + M*j); 
		}
	}
	
}

int main(int argc, char* argv[]) {
	cusolverDnHandle_t cusolverH = NULL;
	cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;

	cudaError_t cudaStat = cudaSuccess;  // cudaSuccess=0, cf. http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#axzz4lEpqZl2L
	
	constexpr const int M = 4; // number of rows 
	constexpr const int N = 2; // number of columns  
	constexpr const int lda = M; 

	cuDoubleComplex A[M*N]; 
	create_linearval_C_matrix(M,N,A); 
	printMatrix(M,N,A,M,"A"); 
	
	cuDoubleComplex U[M*M]; // M-by-M unitary matrix
	cuDoubleComplex VT[N*N]; // N-by-N unitary matrix
	double S[N]; // singular value
	
	cuDoubleComplex *d_A = nullptr; 
	double *d_S = nullptr; 
	cuDoubleComplex *d_U = nullptr; 
	cuDoubleComplex *d_VT = nullptr; 
	int *devInfo = nullptr; 
	cuDoubleComplex *d_work = nullptr; 
	double *d_rwork = nullptr; 
	
	int lwork = 0;
	
	
// step 1: create cusolverDn/cublas handle 
	cusolver_status = cusolverDnCreate(&cusolverH);
	assert (CUSOLVER_STATUS_SUCCESS == cusolver_status);

// step 2: copy A and B to device
	cudaStat = cudaMalloc((void**)&d_A , sizeof(cuDoubleComplex)*M*N);
	assert(cudaSuccess == cudaStat);
	cudaStat = cudaMalloc((void**)&d_S , sizeof(double)*N);
	assert(cudaSuccess == cudaStat);
	cudaStat = cudaMalloc((void**)&d_U , sizeof(cuDoubleComplex)*M*M);
	assert(cudaSuccess == cudaStat);
	cudaStat = cudaMalloc((void**)&d_VT , sizeof(cuDoubleComplex)*N*N);
	assert(cudaSuccess == cudaStat);
	cudaStat = cudaMalloc((void**)&devInfo, sizeof(int));
	assert(cudaSuccess == cudaStat);
	
		
	cudaStat = cudaMemcpy(d_A, A, sizeof(cuDoubleComplex)*M*N, cudaMemcpyHostToDevice);
	assert(cudaSuccess == cudaStat);

// step 3: query working space of SVD 
	cusolver_status = cusolverDnZgesvd_bufferSize(
		cusolverH,
		M,
		N,
		&lwork );
	assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);
	
	cudaStat = cudaMalloc((void**)&d_work , sizeof(cuDoubleComplex)*lwork);
	assert(cudaSuccess == cudaStat);

// step 4: compute SVD 
	signed char jobu = 'A'; // all m columns of U
	signed char jobvt = 'A'; // all n columns of VT
	cusolver_status = cusolverDnZgesvd(
		cusolverH,
		jobu,
		jobvt,
		M,
		N,
		d_A,
		lda,
		d_S,
		d_U,
		M, 	// ldu
		d_VT,
		N, 	// ldvt,
		d_work,
		lwork, 
		d_rwork,
		devInfo);
	cudaStat = cudaDeviceSynchronize();
	assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
	assert(cudaSuccess == cudaStat);
	cudaStat = cudaMemcpy(U,d_U, sizeof(cuDoubleComplex)*lda*M,cudaMemcpyDeviceToHost); 
	assert(cudaSuccess == cudaStat);
	cudaStat = cudaMemcpy(VT,d_VT, sizeof(cuDoubleComplex)*N*N,cudaMemcpyDeviceToHost); 
	assert(cudaSuccess == cudaStat);
	cudaStat = cudaMemcpy(S,d_S, sizeof(double)*N,cudaMemcpyDeviceToHost); 
	assert(cudaSuccess == cudaStat);
	
	
	std::cout << " S = (matlab base-1) " << std::endl; 
	print1darr(N, S, 1, "S");
	std::cout << "=====" << std::endl; 
	
	printf("U = (matlab base-1)\n");
	printMatrix(M, M, U, M, "U");
	printf("=====\n");
	
	printf("VT = (matlab base-1)\n");
	printMatrix(N, N, VT, N, "VT");
	printf("=====\n");
	
	
	


	
// free resources
	if (d_A		) cudaFree(d_A);
	if (d_S		) cudaFree(d_S);
	if (d_U 	) cudaFree(d_U);
	if (d_VT	) cudaFree(d_VT);
	if (devInfo ) cudaFree(devInfo);
	if (d_work	) cudaFree(d_work);
	if (d_rwork	) cudaFree(d_rwork);
	
	if (cusolverH) cusolverDnDestroy(cusolverH);
	
	
	

	cudaDeviceReset();
	return 0;




}
