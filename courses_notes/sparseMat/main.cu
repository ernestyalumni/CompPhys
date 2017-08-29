/**
 * @file   : main.cu
 * @brief  : Sparse Matrices using CUDA Unified Memory, but for Compute Capability 5.X; "main" file (execute the written functions, classes here in this file)  
 * uses CUDA Unified Memory (Management)
 * @author : Ernest Yeung	ernestyalumni@gmail.com
 * @date   : 20170525
 * @ref    :  cf. https://www5.in.tum.de/lehre/vorlesungen/hpc/WS16/tutorial/sparse_02.pdf
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
 * nvcc -std=c++11 -arch='sm_52' main.cu -o main.exe
 * */
#include <iostream>

#include "sparse_Unified.h"  // CSR_MatVecMultiply

constexpr const int K = 8;
constexpr const int N = 4;
constexpr const int N2 = 6; // number of "columns" of matrix A, needed for vector x
__device__ __managed__ float A[K] = { 10.f,20.f,30.f,40.f,50.f,60.f,70.f,80.f};
__device__ __managed__ int IA[N+1] = { 0,2,4,7,8};
__device__ __managed__ int JA[K] = { 0,1,1,3,2,3,4,5};
__device__ __managed__ float x[N2] = {1.f,2.f,3.f,4.f,5.f,6.f}; 
__device__ __managed__ float y[N];

int main(int argc, char* argv[]) {
	for (int idx=0; idx<K; idx++) {
		std::cout << A[idx] << " "; }
	std::cout << std::endl << std::endl;

	for (int idx=0;idx<N+1;idx++) {
		std::cout << IA[idx] << " " ; }

	constexpr const int TILESIZE=2;
	CSR_MatVecMultiply<float><<<dim3((N+TILESIZE-1)/TILESIZE),dim3(TILESIZE)>>>(A,x,y,JA,IA,N);
	cudaThreadSynchronize();
	
		// sanity check : printout
		/* you can only do the following commented out code for Compute Capability 6.X and higher!  
		 * Please help me with obtaining a GTX 1080 Ti by donating at the PayPal link above so I can do so! -EY *//*
/*
	for (int idx=0; idx<N; idx++) {
		std::cout << y[idx] << " "; }
	std::cout << std::endl << std::endl;
*/
	
		// with Compute Capability 5.X you must do this:
	float host_y[N];
	cudaMemcpy(host_y, y, sizeof(float) * N, cudaMemcpyDeviceToHost);  
	std::cout << " y : " << std::endl;
	for (int iidx =0; iidx < N; iidx++) { 
		float y_j = host_y[iidx] ;
		std::cout << y_j << " "; 
	}
	
	std::cout << std::endl << std::endl; 


			// Reset host_y to 0 for the next computation
	for (int iidx =0; iidx < N; iidx++) { 
		host_y[iidx] =0.f;
	}
	cudaMemcpy(y,host_y, sizeof(float) * N, cudaMemcpyHostToDevice);  

	constexpr const int WARPSIZE=32;
	CSR_MatVecMultiply_Warped<float,TILESIZE><<<dim3((N*WARPSIZE+TILESIZE-1)/TILESIZE),dim3(TILESIZE)>>>(A,x,y,JA,IA,N,WARPSIZE);
	cudaThreadSynchronize();
	
	cudaMemcpy(host_y, y, sizeof(float) * N, cudaMemcpyDeviceToHost);  

	std::cout << " y : " << std::endl;
	for (int iidx =0; iidx < N; iidx++) { 
		float y_j = host_y[iidx] ;
		std::cout << y_j << " "; 
	}

	
}

