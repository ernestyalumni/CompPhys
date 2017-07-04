/**
 * @file   : create_MPS_step0.cu
 * @brief  : Step 0 of creating Matrix Product State (MPS) consisting of 2 routines; singular value decomposition (SVD) and matrix multiplication (CUBLAS)
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

void create_fixed_CC_mat(const int d, const int L, cuDoubleComplex *A) {
	int M = 1<<(L-1);  // d^{(L-1)}, where d is dim. of state space, L is number of sites
	for (int i =0; i< M; i++) { // i is the "row" of a matrix, it's an index
		double f = ((double) i*(0.9/M)+0.1);
		double theta_f = 2.* acos( -1.)*f;
		cuDoubleComplex Ad0 = { f*cos(theta_f) ,  f*sin(theta_f) } ; 
		cuDoubleComplex Ad1 = { (1.-f)*sin(theta_f) , (1.-f)*cos(theta_f) } ; 
		A[i + M*0] = Ad0 ; 
		A[i + M*1] = Ad1 ; 
	}
	
}

template <typename TT>
void print1darr(const int N, const TT *A, int lda, const char* name) 
{
	std::cout << name << std::endl;
	for (int row =0; row < N; row++) {
		TT Areg = A[row *lda]; 
		std::cout << Areg.x << "+i"<< Areg.y << " " ; 
//		std::cout << Areg << " ";
	}
	std::cout << std::endl;
}
 
int main(int argc, char* argv[]) {
	
	constexpr const int L = 2; // number of sites
	constexpr const int d = 2; // dim. of state space  
	constexpr const int lda = 1<<(L-1);

	cuDoubleComplex Psi[lda*d]; 
	cuDoubleComplex US[lda*d]; // d^{L-1)-by-d unitary matrix
	cuDoubleComplex VT[d*d]; // d-by-d unitary matrix

	// sanity check
//	double S[d]; // 1-dim. array 
//	cuDoubleComplex S[d]; // 1-dim. array 
//	cuDoubleComplex U[lda*lda];  

		
	cuDoubleComplex *d_Psi = nullptr;
	double *d_S = nullptr;
	cuDoubleComplex *d_SCC = nullptr; // cuDoubleComplex version of S, 1-dim. array of singular values
	cuDoubleComplex *d_U = nullptr; 
	cuDoubleComplex *d_VT = nullptr; 
	int *devInfo = NULL;
	cuDoubleComplex *d_work = nullptr;
	double *d_rwork = NULL;

	// cuBLAS matrix multiplication step
	cuDoubleComplex *d_US = nullptr;  

	int lwork =0;

	/* ************************************************************** */
	/* ************************************************************** */
	/* ****** BOILERPLATE initialization, values ******************** */
	/* ************************************************************** */
	create_fixed_CC_mat(d,L,Psi); 
	printMatrix(1<<(L-1),d,Psi,1<<(L-1),"Psi");

	/* ************************************************************** */
	/* ****** END of BOILERPLATE initialization, values ************* */
	/* ************************************************************** */
	

	cusolverDnHandle_t cusolverH = nullptr;
	cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
// step 1: create cusolverDn handle 
	cusolver_status = cusolverDnCreate(&cusolverH);

	cublasHandle_t cublasH = nullptr;
	cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
// step 1b: create cublas handle 
	cublasCreate(&cublasH);

// step 2: copy A and B to device
	cudaMalloc((void**)&d_Psi , sizeof(cuDoubleComplex)*lda*d);
	cudaMalloc((void**)&d_S , sizeof(double)*d);
	cudaMalloc((void**)&d_SCC , sizeof(cuDoubleComplex)*d);
	cudaMalloc((void**)&d_U , sizeof(cuDoubleComplex)*lda*lda);
	cudaMalloc((void**)&d_VT , sizeof(cuDoubleComplex)*d*d);
	cudaMalloc((void**)&devInfo, sizeof(int));

	cudaMalloc((void**)&d_US , sizeof(cuDoubleComplex)*lda*lda);

	cudaMemcpy(d_Psi, Psi, sizeof(cuDoubleComplex)*lda*d, cudaMemcpyHostToDevice);

// step 3: query working space of SVD 
	cusolver_status = cusolverDnZgesvd_bufferSize(cusolverH,  // cusolver Handle
		lda,d,  // matrix size dimensions of Psi
		&lwork );
	cudaMalloc((void**)&d_work , sizeof(cuDoubleComplex)*lwork);
	
// step 4: compute SVD 
	cusolver_status = cusolverDnZgesvd(cusolverH,'A','A',
		lda,d,
		d_Psi,lda,
		d_S,
		d_U,lda, 	// ldu
		d_VT,d, 	// ldvt,
		d_work,lwork,d_rwork,devInfo);

// change type of 1-dim. array of singular values S from double to cuDoubleComplex
	cudaMemcpy2D(d_SCC, // dst - Destination memory address
					sizeof(cuDoubleComplex), // dpitch - Pitch of destination memory (1 cuDoubleComplex, so skip over 2 double values)
					d_S, 	// src
					1*sizeof(double), 	// spitch
					sizeof(double), 	// width of matrix transfer (columns in bytes)
					d, 					// height of matrix transfer (rows)
					cudaMemcpyDeviceToDevice); 

	cudaDeviceSynchronize();

// Matrix Multiply U*S to obtain US, or new Psi, "flattened".  
	cublas_status = cublasZdgmm(cublasH, CUBLAS_SIDE_RIGHT,
		lda,lda,
		d_U,lda,
		d_SCC,1,
		d_US,lda);


	
	cudaMemcpy(US,d_US, sizeof(cuDoubleComplex)*lda*lda,cudaMemcpyDeviceToHost); 
	cudaMemcpy(VT,d_VT, sizeof(cuDoubleComplex)*d*d,cudaMemcpyDeviceToHost); 

// sanity check
//	cudaMemcpy(S,d_S, sizeof(double)*d,cudaMemcpyDeviceToHost); 
//	cudaMemcpy(S,d_SCC, sizeof(cuDoubleComplex)*d,cudaMemcpyDeviceToHost); 
//	cudaMemcpy(U,d_U, sizeof(cuDoubleComplex)*lda*lda,cudaMemcpyDeviceToHost); 
	

	std::cout << "VT = (matlab base-1) " << std::endl; 
	printMatrix(d, d, VT, d, "VT");
	std::cout << "===== " << std::endl;


// sanity check
//	print1darr<double>(d,S,1," S ");
//	print1darr<cuDoubleComplex>(d,S,1," S ");
//	printMatrix(lda, lda, U, lda, "U");

	/* 
	 * I will demonstrate here the reshaping of the d^{L-1}xd matrix US into a 
	 * d^{L-2}xdr_{L-1} matrix Psi' by showing, in terms of the practical implementation in C/C++/CUDA, 
	 * that it's only a change in "stride" 
	 * */

	// sanity check
//	cudaDeviceSynchronize();
//	std::cout << US[0].x <<"+i"<<US[0].y<<" " << US[1].x <<"+i"<<US[1].y<<" "<< US[2].x <<"+i"<<US[2].y<<" "<< US[3].x <<"+i"<<US[3].y<<" "<< std::endl;
	
//	std::cout << " 1 <<(L-2) : " << (1<<(L-2) ) << std::endl;  // 1
	 for (int Inew =0; Inew < (1<<(L-2)); Inew++) // Inew is I_{L-2}, i.e. the index (\sigma_0 ... \sigma_{L-2}), index after reshaping
	 {
		 for (int a_Lm2 =0; a_Lm2 < d*d; a_Lm2++) { 
			int a_Lm1 = a_Lm2/d;  // integer division of a_Lm2 by d should yield a_{L-1}, index for state of site L-1
			int sigma_Lm2 = a_Lm2 % d; // mod, or remainder after division by d should give \sigma_{L-2}, index of which state site L-2 is in 
			int I_Lm1 = Inew + (1<<(L-2))*sigma_Lm2;
			
			// with I_Lm1, and a_Lm1, we can access the original entries/values in US, before it becomes reshaped into the new matrix Psi
			cuDoubleComplex US_og = US[ I_Lm1 + (1<<(L-1))* a_Lm1];
			std::cout << US_og.x << "+i" << US_og.y << " ";
		}
		std::cout << std::endl; 
	}
	
	
// free resources
	if (d_Psi		) cudaFree(d_Psi);
	if (d_S		) cudaFree(d_S);
	if (d_SCC		) cudaFree(d_SCC);

	if (d_U 	) cudaFree(d_U);
	if (d_VT	) cudaFree(d_VT);
	if (devInfo ) cudaFree(devInfo);
	if (d_work	) cudaFree(d_work);
	if (d_rwork	) cudaFree(d_rwork);
	
	if (d_US	) cudaFree(d_US);
		
	if (cusolverH) cusolverDnDestroy(cusolverH);
	if (cublasH	) cublasDestroy(cublasH);
	
	
	

	cudaDeviceReset();
	return 0;



}

