/**
 * @file   : SVD_eg_unified.cu
 * @brief  : More examples of singular value decomposition of a real matrix  
 * 				compute A = U*S*VT
 * @author : Ernest Yeung	ernestyalumni@gmail.com
 * @date   : 20170701
 * @ref    :  cf. https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-241j-dynamic-systems-and-control-spring-2011/readings/MIT6_241JS11_chap04.pdf
 * https://www.mathworks.com/content/dam/mathworks/mathworks-dot-com/moler/eigs.pdf
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
 * EY : 20170628 This worked for the 1st example, but not the second.  
 * nvcc -std=c++11 -arch='sm_52' -lcudart -lcublas -lcusolver SVD_eg_unified.cu -o SVD_eg_unified.exe
 * */
 
#include <iostream> 	// std::cout
#include <iomanip> 		// std::setprecision 

#include <cusolverDn.h> // Dn = dense (matrices)

__device__ __managed__ double A[2*2]; 
__device__ __managed__ double U[2*2]; 
__device__ __managed__ double VT[2*2]; 
__device__ __managed__ double S[2]; 
__device__ __managed__ int *devInfo = nullptr; 

// for a small example, 10.2 of Ch. 10 Eigenvalues and Singular Values of 
// cf. https://www.mathworks.com/content/dam/mathworks/mathworks-dot-com/moler/eigs.pdf
__device__ __managed__ double A2[3*3]; 
__device__ __managed__ double U2[3*3]; 
__device__ __managed__ double VT2[3*3]; 
__device__ __managed__ double S2[3]; 
__device__ __managed__ int *devInfo2 = nullptr; 



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

	// --- cuSOLVER input/output parameters/arrays
	// working space, <type> array of size lwork
	double *d_work = NULL;
	// size of working array work
	int lwork = 0;

// step 1: create cusolverDn/cublas handle, CUDA solver initialization
	cusolver_status = cusolverDnCreate(&cusolverH);

	// --- Setting the boilerplate, initialization values
	std::cout << "\n Examples from elsewhere : " << std::endl; 
	A[0] = 100. ;
	A[1] = 100.2 ;
	A[2] = 100.;
	A[3] = 100.;
	printMatrix(2,2,A,2,"A");
	printMatrix(2,2,U,2,"U");
	printMatrix(2,1,S,1,"S");
	printMatrix(2,2,VT,2,"VT");


// step 2: query working space of SVD	
	cusolver_status = cusolverDnDgesvd_bufferSize( cusolverH, 2, 2, &lwork);
	cudaMalloc((void**)&d_work, sizeof(double)*lwork);

// step 4: compute SVD

	// --- CUDA SVD execution
	cusolver_status = cusolverDnDgesvd(cusolverH,'A', 'A', 2,2, A, 2, 
		S,U, 
		2, // ldu
		VT, 
		2, // ldvt
		d_work,lwork, NULL, devInfo);

	printMatrix(2,2,A,2,"A");
	printMatrix(2,2,U,2,"U");
	printMatrix(2,1,S,1,"S");
	printMatrix(2,2,VT,2,"VT");


// 2nd Example from mathworks
// cf. https://www.mathworks.com/content/dam/mathworks/mathworks-dot-com/moler/eigs.pdf
	cusolverDnHandle_t cusolverH2 = NULL;

	// --- cuSOLVER input/output parameters/arrays
	// working space, <type> array of size lwork
	double *d_work2 = NULL;
	// size of working array work
	int lwork2 = 0;

// step 1: create cusolverDn/cublas handle, CUDA solver initialization
	cusolver_status = cusolverDnCreate(&cusolverH2);

	// --- Setting the boilerplate, initialization values
	std::cout << "\n Examples from elsewhere : " << std::endl; 
	A2[0] = -149. ;
	A2[1] = 537. ;
	A2[2] = -27.;
	A2[3] = -50.;
	A2[4] = 180.;
	A2[5] = -9.;
	A2[6] = -154.;
	A2[7] = 546.;
	A2[8] = -25.;
	printMatrix(3,3,A2,3,"A2");

	
// step 2: query working space of SVD	
	cusolver_status = cusolverDnDgesvd_bufferSize( cusolverH2, 3, 3, &lwork2);

	cudaMalloc((void**)&d_work2, sizeof(double)*lwork2);


// step 4: compute SVD
	// --- CUDA SVD execution
	cusolver_status = cusolverDnDgesvd(cusolverH2,'A', 'A', 3,3, A2, 3, 
		S2,U2, 
		3, // ldu
		VT2, 
		3, // ldvt
		d_work2,lwork2, NULL, devInfo2);

	cudaDeviceSynchronize();


	printMatrix(3,3,U2,3,"U2");
	printMatrix(3,1,S2,1,"S2");
	printMatrix(3,3,VT2,3,"VT2");


// free resources
	if (cusolverH) cusolverDnDestroy(cusolverH);
	if (d_work) cudaFree(d_work);


// free resources
	if (cusolverH2) cusolverDnDestroy(cusolverH2);
	if (d_work2) cudaFree(d_work2);

	
	cudaDeviceReset();
	return 0;


}
