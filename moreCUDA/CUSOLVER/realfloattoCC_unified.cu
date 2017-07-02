/**
 * @file   : realfloattoCC_unified.cu
 * @brief  : Simple Examples of taking an array of (real) float to complex numbers (CC)    
 * this implementation uses CUDA Unified Memory Management
 *
 * @author : Ernest Yeung	ernestyalumni@gmail.com
 * @date   : 20170701
 * @ref    :  https://stackoverflow.com/questions/13535182/copying-data-to-cufftcomplex-data-struct
 * http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g3a58270f6775efe56c65ac47843e7cee
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
 * nvcc -std=c++11 -arch='sm_52' realfloattoCC_unified.cu -o realfloattoCC_unified.exe
 * */

#include <iostream> 	// std::cout

#include <cuComplex.h> // cuComplex, cuDoubleComplex

#include <assert.h> // assert

constexpr const int N = 4;
constexpr const int N2 = 10; 

__device__ __managed__ float realx[N];
__device__ __managed__ float2 vec2_xy[N];
__device__ __managed__ cuComplex CCz[N];

__device__ __managed__ double doublex[N];
__device__ __managed__ cuDoubleComplex doubleCCz[N];

// for a second example, repetition
__device__ __managed__ float realx2[N2];
__device__ __managed__ float2 vec2_xy2[N2];
__device__ __managed__ cuComplex CCz2[N2];

__device__ __managed__ double doublex2[N2];
__device__ __managed__ cuDoubleComplex doubleCCz2[N2];


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
	
	cudaError_t cudaStat = cudaSuccess; // cudaSuccess=0, cf. http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#axzz4lEpqZl2L

	// --- boilerplate initialization  
	int ind=11;
	for (int i=0; i<N; i++) {
		realx[i] = (float) ind++;
		doublex[i] = (double) (i+1);
	}
	print1darr<float>(N,realx,1,"realx");  std::cout << std::endl; 
	print1darr<double>(N,doublex,1,"doublex");  std::cout << std::endl; 

	
	/* 
	 * I'd like to reiterate, this really helped with understanding how to utilize cudaMemcpy2D in such a useful manner: 
	 * https://stackoverflow.com/questions/13535182/copying-data-to-cufftcomplex-data-struct
	 * 
	 * */
	 /* 
	  * Copy real values that represent the real part of a complex number, to an array of complex numbers
	  * the imaginary part should all still be 0, i.e.
	  * 
	  * (11, 12,13,14) -> (11 + i*0, 12 + i*0, 13 + i*0, 14 + i*0 )  
	  * */
	  
	cudaStat = cudaMemcpy2D( vec2_xy, 			// dst - Destination memory address  
							 2*sizeof(float), 	// dpitch - Pitch of destination memory  
							 realx, 			// src - Source memory address  
							1*sizeof(realx[0]), 	// spitch - Pitch of source memory; in this case realx[0] is a float
							sizeof(realx[0]), 		// width of matrix transfer (columns in bytes); in this case realx[0] is a float
							N,						// height of matrix transfer (rows)
							cudaMemcpyDeviceToDevice);  
	/* 
	 * Remarks:
	 * It appears here and in the CUDA Toolkit v8 Documentation, API, that row and columns usage assume a ROW-MAJOR ordering (bunch up "column" values first, before rows)
	 * 
	 * */
	
	cudaDeviceSynchronize();  
	
	// sanity check
	print2darr<float2>(N, vec2_xy, 1, "vec2_xy");

	// more sanity check
	std::cout << " sizeof(float2) " << sizeof(float2) << std::endl; 
	std::cout << " sizeof(cuComplex) " << sizeof(cuComplex) << std::endl; 

	/*
	 * Now try copying real numbers, but represent the imaginary part, to that array of complex numbers, i.e.
	 * (11,12,13,14) -> (0 + i*11, 0+i*12,0+i*13,0+i*14)
	 * 
	 * For why we need this temporary pointer tmp_CCz, see 
	 * https://stackoverflow.com/questions/13535182/copying-data-to-cufftcomplex-data-struct
	 * otherwise my explanation is that cuComplex CCz[N], as a pointer, assumes that when we move or increment by 1 the 
	 * pointer itself, we increment by a float2 or cuComplex or 2 floats over, skipping over the "imaginary" part
	 * this temporary pointer, which points to the same thing, now increments by 1 to mean incrementing by a single float
	 * */

	float* tmp_CCz = (float *) CCz; 
	cudaStat = cudaMemcpy2D( tmp_CCz + 1, 			// dst - 
							sizeof(cuComplex), // dpitch - Pitch of destination memory (2*float, so skip over 1 float)
							realx, 				// src
							1*sizeof(realx[0]),	// spitch
							sizeof(realx[0]), 	// width of matrix transfer (columns in bytes)
							N,					// height of matrix transfer (rows) 
							cudaMemcpyDeviceToDevice);  
							
	cudaDeviceSynchronize();  
	// sanity check
	print2darr<float2>(N, CCz, 1, "CCz"); // should get 0 11 0 12 0 13 0 14 
 	
	// works with doubles  
	cudaStat = cudaMemcpy2D( doubleCCz, 2*sizeof(double), 
							 doublex, 1*sizeof(doublex[0]), 
							 sizeof(doublex[0]), N, 
							 cudaMemcpyDeviceToDevice);
	cudaDeviceSynchronize();  
	// sanity check
	print2darr<cuDoubleComplex>(N, doubleCCz, 1, "doubleCCz"); // should get 1 0 2 0 3 0 4 0 
							 
							
	
}


//cudaMemcpy2D( 
