/**
 * @file   : create_MPS_step1_large.cu
 * @brief  : 2 Steps of creating Matrix Product State (MPS) consisting of 2 routines; singular value decomposition (SVD) and matrix multiplication (CUBLAS)
 * 				This is an example for large L (L=sites), so 2^L states possible
 * @author : Ernest Yeung	ernestyalumni@gmail.com
 * @date   : 20170704
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

#include <array> // std::array

#include <cuda_runtime.h>  // cudaError_t
#include <cusolverDn.h> // Dn = dense (matrices)

#include <cuComplex.h> // cuComplex, cuDoubleComplex

#include "gputimer.h" // GpuTimer



/* ****************************************************************** */
/* ****** "BOILERPLATE" routines for creating arbitrary initialization 
 * values, print out for human reaidng of results *********************
/* ****************************************************************** */

void printMatrix(int m, int n, const cuDoubleComplex *A, int lda, const char* name) 
{
	std::cout << name << std::endl;
	for (int row =0; row <m; row++) {
		for (int col =0 ; col <n ; col++) {
			cuDoubleComplex Areg = A[row + col*lda]; 
			std::cout << std::setprecision(5) << Areg.x << "+i" << Areg.y << " " ; 
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
		std::cout << Areg.x << "+i"<< Areg.y << " " ; 
//		std::cout << Areg << " ";
	}
	std::cout << std::endl;
}

/* ****************************************************************** */
/* ****** END of boilerplate ******************************************/
/* ****************************************************************** */

/* ****************************************************************** */
/* ****** MAIN routine ************************************************/
/* ****************************************************************** */


int main(int argc, char* argv[]) {
	
	constexpr const int L = 14; // number of sites
	constexpr const int d = 2; // dim. of state space  
	int lda = 1<<(L-1);

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
	
	// calculate new Psi
	cuDoubleComplex *d_Psi_new = nullptr;


	int lwork =0;

	/* ************************************************************** */
	/* ************************************************************** */
	/* ****** BOILERPLATE initialization, values ******************** */
	/* ************************************************************** */
//	create_fixed_CC_mat(d,L,Psi);
//	printMatrix(lda,d, &Psi.data() ,lda,"Psi");
//	cuDoubleComplex Psi[lda*d]; 
//	cuDoubleComplex*  Psi = new cuDoubleComplex[lda*d]; 
	// boilerplate, initialization; matrix assumed to be column-major ordering 
	std::array<cuDoubleComplex, (1<<(L-1))*d> Psi;
	{
		int M = 1<<(L-1);  // d^{(L-1)}, where d is dim. of state space, L is number of sites
		for (int i =0; i< M; i++) { // i is the "row" of a matrix, it's an index
			double f = ((double) i*(0.9/M)+0.1);
			double theta_f = 2.* acos( -1.)*f;
			cuDoubleComplex Ad0 = { f*cos(theta_f) ,  f*sin(theta_f) } ; 
			cuDoubleComplex Ad1 = { (1.-f)*sin(theta_f) , (1.-f)*cos(theta_f) } ; 
			Psi[i + M*0] = Ad0 ; 
			Psi[i + M*1] = Ad1 ; 
			std::cout << std::setprecision(5) << Ad0.x << "+i" << Ad0.y << " " << Ad1.x << "+i" << Ad1.y << std::endl; 
		}
	}


	GpuTimer timer;
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

	cudaMalloc((void**)&d_Psi_new , sizeof(cuDoubleComplex)*lda*d);


	cudaMemcpy(d_Psi, Psi.data(), sizeof(cuDoubleComplex)*lda*d, cudaMemcpyHostToDevice);

// step 3: query working space of SVD 
	timer.Start();  // timer "boilerplate"

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

// Matrix Multiply U*S to obtain US, or new Psi, "flattened".  
	cublas_status = cublasZdgmm(cublasH, CUBLAS_SIDE_RIGHT,
		lda,lda,
		d_U,lda,
		d_SCC,1,
		d_US,lda);

// obtain new Psi, 1st step - "reduce" matrix size dim. to the Schmidt rank  
	cudaMemcpy(d_Psi_new, d_US, sizeof(cuDoubleComplex)*lda*d, cudaMemcpyDeviceToDevice);

// sanity check  
//	cuDoubleComplex US[lda*lda]; // d^{L-1)-by-d unitary matrix
	cuDoubleComplex VT[d*d]; // d-by-d unitary matrix
	cudaMemcpy(VT,d_VT, sizeof(cuDoubleComplex)*d*d,cudaMemcpyDeviceToHost); 
//	cudaMemcpy(US,d_US, sizeof(cuDoubleComplex)*lda*lda,cudaMemcpyDeviceToHost); 	
//	printMatrix(lda, lda, US, lda, "US"); // 1 should observe many 0 entries which is expected
	printMatrix(d, d, VT, d, "VT");
//	cuDoubleComplex Psi_new[lda*d]; // d^{L-1)-by-d unitary matrix
//	cudaMemcpy(Psi_new,d_Psi_new, sizeof(cuDoubleComplex)*lda*d,cudaMemcpyDeviceToHost); 	
//	printMatrix(lda, d, Psi_new, lda, "Psi_new");




// free resources
	if (d_Psi		) cudaFree(d_Psi);
	if (d_S		) cudaFree(d_S);
	if (d_SCC		) cudaFree(d_SCC);

	if (d_U 	) cudaFree(d_U);
	if (d_US	) cudaFree(d_US);
	if (d_work	) cudaFree(d_work);


/* ****************************************************************** */
/* ****** 2nd iteration ********************************************* */
/* ****************************************************************** */
 
	double *d_S_l2 = nullptr;
	cuDoubleComplex *d_SCC_l2 = nullptr; // cuDoubleComplex version of S, 1-dim. array of singular values
	cuDoubleComplex *d_U_l2 = nullptr; 
	cuDoubleComplex *d_VT_l2 = nullptr; 
	cuDoubleComplex *d_work_l2 = nullptr;

	// cuBLAS matrix multiplication step
	cuDoubleComplex *d_US_l2 = nullptr;  
	
	// calculate new Psi
	cuDoubleComplex *d_Psi_new_l2 = nullptr;

// step 2: device memory (GPU) allocation
	lda = (1<<(L-2)); 
	const int dr = d*d;

// sanity check
//	cuDoubleComplex Psi_new[lda*dr];  
//	cudaMemcpy(Psi_new,d_Psi_new, sizeof(cuDoubleComplex)*lda*lda,cudaMemcpyDeviceToHost); 
//	printMatrix(lda, lda, Psi_new, lda, "Psi_new");



	cudaMalloc((void**)&d_S_l2 , sizeof(double)*dr*dr);
	cudaMalloc((void**)&d_SCC_l2 , sizeof(cuDoubleComplex)*dr*dr);
	cudaMalloc((void**)&d_U_l2 , sizeof(cuDoubleComplex)*lda*lda);
	cudaMalloc((void**)&d_VT_l2 , sizeof(cuDoubleComplex)*dr*dr);

	cudaMalloc((void**)&d_US_l2 , sizeof(cuDoubleComplex)*lda*lda);

	cudaMalloc((void**)&d_Psi_new_l2 , sizeof(cuDoubleComplex)*lda*dr);

// step 3: query working space of SVD 
	cusolver_status = cusolverDnZgesvd_bufferSize(cusolverH,  // cusolver Handle
		lda,dr,  // matrix size dimensions of Psi
		&lwork );
	cudaMalloc((void**)&d_work_l2 , sizeof(cuDoubleComplex)*lwork);

// step 4: compute SVD 
	cusolver_status = cusolverDnZgesvd(cusolverH,'A','A',
		lda,dr,
		d_Psi_new,lda,
		d_S_l2,
		d_U_l2,lda, 	// ldu
		d_VT_l2,dr, 	// ldvt,
		d_work_l2,lwork,NULL,devInfo);

// change type of 1-dim. array of singular values S from double to cuDoubleComplex
	cudaMemcpy2D(d_SCC_l2, // dst - Destination memory address
					sizeof(cuDoubleComplex), // dpitch - Pitch of destination memory (1 cuDoubleComplex, so skip over 2 double values)
					d_S_l2, 	// src
					1*sizeof(double), 	// spitch
					sizeof(double), 	// width of matrix transfer (columns in bytes)
					dr, 					// height of matrix transfer (rows)
					cudaMemcpyDeviceToDevice); 

// sanity check
//	cuDoubleComplex S_l2[dr]; // 1-dim. array 
//	cuDoubleComplex U_l2[lda*lda];  
//	cudaMemcpy(S_l2,d_SCC_l2, sizeof(cuDoubleComplex)*dr,cudaMemcpyDeviceToHost); 
//	cudaMemcpy(U_l2,d_U_l2, sizeof(cuDoubleComplex)*lda*lda,cudaMemcpyDeviceToHost); 
//	print1darr<cuDoubleComplex>(dr,S_l2,1," S_l2 ");
//	printMatrix(lda, lda, U_l2, lda, "U_l2");

	
// Matrix Multiply U*S to obtain US, or new Psi, "flattened".  
	cublas_status = cublasZdgmm(cublasH, CUBLAS_SIDE_RIGHT,
		lda,lda,
		d_U_l2,lda,
		d_SCC_l2,1,
		d_US_l2,lda);

// obtain new Psi, 1st step - "reduce" matrix size dim. to the Schmidt rank  
	cudaMemcpy(d_Psi_new_l2, d_US_l2, sizeof(cuDoubleComplex)*lda*dr, cudaMemcpyDeviceToDevice);


	// timer "boilerplate"
	timer.Stop();


	cudaDeviceSynchronize();

// sanity check  
//	cuDoubleComplex US_l2[lda*lda]; // d^{L-2)-by-dr unitary matrix
//	cudaMemcpy(US_l2,d_US_l2, sizeof(cuDoubleComplex)*lda*lda,cudaMemcpyDeviceToHost); 	
//	printMatrix(lda, lda, US_l2, lda, "US_l2 (2nd iteration)");
//	cuDoubleComplex Psi_new_l2[lda*dr]; // d^{L-1)-by-d unitary matrix
	std::array<cuDoubleComplex, (1<<(L-2))*dr> Psi_new_l2;
	cudaMemcpy(Psi_new_l2.data(),d_Psi_new_l2, sizeof(cuDoubleComplex)*lda*dr,cudaMemcpyDeviceToHost); 	
	{
	for (int row =0; row <lda; row++) {
		for (int col =0 ; col <dr ; col++) {
			cuDoubleComplex Areg = Psi_new_l2[row + col*lda]; 
			std::cout << std::setprecision(5) << Areg.x << "+i" << Areg.y << " " ; 
		}
		std::cout << std::endl;
	}
}
//	cudaDeviceSynchronize();
//	printMatrix(lda, dr, Psi_new_l2, lda, "Psi_new_l2 (2nd iteration)");
//	std::cout << " lda for 2nd iteration : " << lda << std::endl; // sanity check
//	std::cout << " dr for 2nd iteration : " << dr << std::endl; // sanity check for matrix size dim.

	cuDoubleComplex VT_l2[dr*dr]; // d^{L-1)-by-d unitary matrix
	cudaMemcpy(VT_l2,d_VT_l2, sizeof(cuDoubleComplex)*dr*dr,cudaMemcpyDeviceToHost); 	
	std::cout << "\n VT = (matlab base-1), 2nd. iteration " << std::endl; 
	printMatrix(dr, dr, VT_l2, dr, "VT 2nd iteration");
	std::cout << "===== " << std::endl;
	

//	std::cout << " Calculation of 2 iterations of SVD and matrix multiplication  : " << std::setprecision(7) << timer.Elapsed() << " ms " << 
	std::cout << " Calculation of 2 iterations of SVD and matrix multiplication  : " << timer.Elapsed() << " ms " << 
		" for " << (1<<L) << " states (of the system " << std::endl;


// free all resources
	if (d_VT	) cudaFree(d_VT);
	if (d_VT_l2	) cudaFree(d_VT_l2);

	if (d_U_l2	) cudaFree(d_U_l2);
	if (d_work_l2	) cudaFree(d_work_l2);

	if (devInfo ) cudaFree(devInfo);

	if (d_rwork	) cudaFree(d_rwork);
	
	if (d_US_l2	) cudaFree(d_US_l2);
	if (d_Psi_new		) cudaFree(d_Psi_new);
	if (d_Psi_new_l2		) cudaFree(d_Psi_new_l2);

		
	if (cusolverH) cusolverDnDestroy(cusolverH);
	if (cublasH	) cublasDestroy(cublasH);
	
	cudaDeviceReset();
	return 0;

}
