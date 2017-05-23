/**
 * @file   : main_matmultShared_Unified.cu
 * @brief  : Matrix Multiplication using Shared Memory and CUDA Unified Memory, but for Compute Capability 5.X  
 * uses CUDA Unified Memory (Management)
 * @author : Ernest Yeung	ernestyalumni@gmail.com
 * @date   : 20170522
 * @ref    :  cf. based on code from CUDA C Programming Guide
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
 * P.S. I'm using an EVGA GeForce GTX 980 Ti which has at most compute capability 5.2; 
 * A hardware donation or financial contribution would help with obtaining a 1080 Ti with compute capability 6.2
 * so that I can do -arch='sm_62' and use many new CUDA Unified Memory features
 * */
/**
 * COMPILATION TIP
 * 
 * nvcc -std=c++11 -arch='sm_61' main_matmultShared_Unified.cu -o main_matmultShared_Unified.exe
 * 
 * */
#include <iostream>


constexpr const int NA = 4;
constexpr const int NB = 8; 
constexpr const int NC = 6;
constexpr const int BLOCK_SIZE = 2;

__device__ __managed__ float A[NA*NB];
__device__ __managed__ float B[NB*NC];
__device__ __managed__ float C[NA*NC];
 

// Matrix multiplication kernel
template <typename Type, int BLOCK_SIZE>
__global__ void MatMulkernel(Type* A, Type* B, Type* C, 
							 const int NA, const int NB, const int NC) {
	int blockI = blockIdx.y;
	int blockJ = blockIdx.x;
	
	Type Cvalue = ((Type) 0.f);
	
	int i = threadIdx.y;
	int j = threadIdx.x;
	
	for (int j_K =0; j_K < NB/BLOCK_SIZE; ++j_K) { 
		
		__shared__ float Ash[BLOCK_SIZE][BLOCK_SIZE] ;
		__shared__ float Bsh[BLOCK_SIZE][BLOCK_SIZE] ;

		int idxA = j_K*BLOCK_SIZE + j; // "j global index"
		idxA += NB*(BLOCK_SIZE*blockI + i);
		
		Type Aval = A[idxA];
			
		Ash[i][j] = Aval; 
	
		int idxB = blockJ*BLOCK_SIZE + j; // "j global index"
		idxB += NC*(BLOCK_SIZE*j_K + i);
		
		Type Bval = B[idxB];
		
		Bsh[i][j] = Bval; 
	
		__syncthreads();
		
		for (int k=0;k<BLOCK_SIZE;++k) {
			Cvalue += Ash[i][k]*Bsh[k][j]; }
			
		__syncthreads();
	}
	
	int idxC = blockJ*BLOCK_SIZE + j;
	idxC += NC*(BLOCK_SIZE*blockI + i) ;
	
	C[idxC] = Cvalue;
	
}

int main() {
	// boilerplate
	for (int iidx=0;iidx<NA;iidx++) {
		for (int jidx=0;jidx<NB;jidx++) {
			if (jidx>=iidx) {
				A[ iidx*NB + jidx] = 1.f; }
			else {
				A[ iidx*NB+jidx] = 1.f; }
		}
	}

	for (int iidx=0;iidx<NB;iidx++) {
		for (int jidx=0;jidx<NC;jidx++) {
			if (jidx>=iidx) {
				B[ iidx*NC + jidx] = 2.f; }
			else {
				B[ iidx*NC+jidx] = 3.f; }
		}
	}

	
		// sanity check: print out
	std::cout << " A : " << std::endl;
	for (int iidx=0;iidx<NA;iidx++) {
		for (int jidx=0;jidx<NB;jidx++) {
			std::cout << A[iidx*NB+jidx] << " ";
		}
		std::cout << std::endl;
	} std::cout << std::endl;

	std::cout << " B : " << std::endl;
	for (int iidx=0;iidx<NB;iidx++) {
		for (int jidx=0;jidx<NC;jidx++) {
			std::cout << B[iidx*NC+jidx] << " ";
		}
		std::cout << std::endl;
	} std::cout << std::endl;


	MatMulkernel<float,
				BLOCK_SIZE><<<dim3(NC/BLOCK_SIZE,NA/BLOCK_SIZE),
								dim3(BLOCK_SIZE,BLOCK_SIZE)>>>(A,B,C,NA,NB,NC);
//	cudaThreadSynchronize();
	cudaDeviceSynchronize();
		
		// sanity check: print out
	std::cout << " C : " << std::endl;
	for (int iidx=0;iidx<NA;iidx++) {
		for (int jidx=0;jidx<NC;jidx++) {
			std::cout << C[iidx*NC+jidx] << " ";
		}
		std::cout << std::endl;
	} std::cout << std::endl;
	
	
}
