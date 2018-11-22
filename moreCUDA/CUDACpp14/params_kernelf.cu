/**
 * @file   : params_kernelf.cu
 * @brief  : Modified implementation of njuffa's; 
 * 				CUDA kernel functions as parameters with CUDA C++14, CUDA Unified Memory Management
 * @details : Modified implementation of njuffa's,  
 * 				std::function vs. function pointer in C++11, C++14, and now in CUDA
 * 				std::function vs. function pointer for CUDA kernel functions (i.e. __global__ )
 * 
 * @author : Ernest Yeung <ernestyalumni@gmail.com>
 * @date   : 20171020  
 * @ref    : https://devtalk.nvidia.com/default/topic/487190/kernel-functions-as-parameters-/
 * https://stackoverflow.com/questions/16498969/how-do-i-typedef-a-function-pointer-with-the-c11-using-syntax
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
 * COMPILATION TIP
 * nvcc -std=c++14 params_kernelf.cu -o params_kernelf.exe
 * 
 * */
#include <iostream>
#include <functional> 	// std::function
#include <type_traits> 	// std::add_pointer

#include <map>			// std::map


#include <stdio.h>

#include <vector> // std::vector

#include <memory> // std::unique_ptr

#define BLOCK_COUNT  240
#define THREAD_COUNT 128

/** @brief function pointer example, pf = "processing function"
 * @ref Scott Meyers, Effective Modern C++, pp. 63, Item 9
 * @details pointer to a function of 2 floats
 * */
// this WORKS
// using pf = float (*)(float, float);

using pf = std::add_pointer<float(float,float)>::type;

__device__ float minimum(float a, float b)
{
    return fminf(a,b);
};

__device__ float maximum(float a, float b)
{
    return fmaxf(a,b);
};

/**
 * @ref http://www.cplusplus.com/forum/beginner/4844/
 * 
 * @details From jsmith 
 * The ! operator is the logical-NOT operator.  
 * In C/C++, integers can be implicitly cast to boolean values according to the rule that 0 is false and anything else is true. The boolean false has "value" 0 and true has "value" 1.
 * Programmers sometimes take advantage of the implicit boolean conversion to perform the conversion:
 * f(x) = { 0, iff x == 0, and 1 iff x != 0 }
 * So given an integer x, !!x is 1 iff x is not zero and 0 otherwise.
 * 
 * */

/**
 * @brief 
 * @ref Scott Meyers, Effective Modern C++, pp. 63, Item 9
*/
__device__ pf func_d[2] = { maximum, minimum };


/* this will be inputted in with the following parameters  
 * x = x_d \in \mathbb{R}^n
 * res = res_d \in \mathbb{R}^{N_x}  
 * n = n = total length or size of the problem array  
 * pf func = func_d[findmin]
 */
__device__ void minmax(float *x, float *res, int n, pf func)
{
	int Mx = blockDim.x; // "THREAD_COUNT" or number of threads in a (single) block, in x-direction

//	__shared__ float partExtr[Mx];  // error: expression must have a constant value
 	__shared__ float partExtr[THREAD_COUNT];  // error: expression must have a constant value
  
    int i;	
    int tid = threadIdx.x; // 0,1,... Mx-1

	int kx = threadIdx.x + blockDim.x*blockIdx.x; // kx=0,1,.. Mx*Nx-1

	int Nx = gridDim.x ; 


    float extr = x[0];

	// i=kx, kx + Nx*Mx, kx + 2*Nx*Mx, ... < n 
	// the big idea is to make sure all values of x=x[i], i=0,1,...n-1, gets calculated
	// this is because we worry about the case when n > Nx*Mx (or perhaps n >> Nx*Mx !!!)
    for (i = kx; i < n; i += Nx*Mx) {
        extr = func(extr, x[i]);
    }
	// tid \in 0,1,...Mx-1; kx = tid + blockDim.x*blockIdx.x
    partExtr[tid] = extr;

	// Mx >> 1 = Mx/2; i= Mx/2, Mx/4, ... 2,1
    for (i = Mx >> 1; i > 0; i >>= 1) {
        __syncthreads(); 
		
		// tid \in 0,1,...Mx-1
		// assuming Mx % 2 = 0, 
		// i= Mx/2, Mx/4 ... 2,1
        if (tid < i) {
			// tid \in 0,1,...Mx-1
			// tid+i, so i < tid+i and Mx/2 < (tid + Mx/2), Mx/4 < (tid+ Mx/4) ... 1 < (tid +1)
			// i.e. we're grabbing the "other half", computing against the "other half" in a 
			// thread block, each time we're accessing tid+i
            partExtr[tid] = func(partExtr[tid], partExtr[tid+i]);
        }
    }

    if (tid == 0) {
        res[blockIdx.x] = partExtr[tid];
    }
}

__global__ void minmax_kernel(float *x, float *res, int n, int findmin)
{
    minmax(x, res, n, func_d[findmin]);
}

/**
 *	@fn findExtremum
 * 	@param int findmin - 0 for fmaxf, 1 for fminf
 * 
 * */
float findExtremum(float *x, int n, int findmin)
{
    pf func_h[2] = { fmaxf, fminf };

    float *res_d;	// result of minmax on device GPU; res_d \in \mathbb{R}^{ N_x }
    //float *res_h;	// result of minmax on host CPU
	std::vector<float> h_resultvec( BLOCK_COUNT, 0.f);  // h_resultvec \in \mathbb{R}^{ N_x}
    float *x_d;		// x array on device GPU, of size n, x \in \mathbb{R}^n
    float r;

    if (n < 1) return sqrtf(-1.0f); // NaN

    cudaMalloc((void**)&res_d, BLOCK_COUNT*sizeof(res_d[0]));
    cudaMalloc((void**)&x_d, n * sizeof(x_d[0]));

    cudaMemcpy(x_d, x, n * sizeof(x_d[0]), cudaMemcpyHostToDevice);

	// ! is logical NOT, and so !! makes a boolean, which is just 0 or 1
    minmax_kernel<<<BLOCK_COUNT,THREAD_COUNT>>>(x_d, res_d, n, !!findmin);
/*
    if (!res_h) {
		fprintf(stderr, "res_h allocation failed\n");
		exit(EXIT_FAILURE);
    }
*/
    cudaMemcpy( h_resultvec.data(), res_d, BLOCK_COUNT * sizeof(res_d[0]), cudaMemcpyDeviceToHost);

    cudaFree(res_d);
    cudaFree(x_d);

    r = h_resultvec[0];

	// i=1,2,...N_x-1; and h_resultvec \in \mathbb{R}^{N_x}
    for (int i = 1; i < BLOCK_COUNT; i++) {
		// int findmin =0,1; func_h[0]=fmaxf, func_h[1]=fminf
		r = func_h[findmin](r, h_resultvec[i]);
	}


    return r;
}

/* =============== sunburntfish's increment, decrement test function =============== */
// @ref https://stackoverflow.com/questions/25848690/should-i-use-stdfunction-or-a-function-pointer-in-c
// @ref pp. 39, Item 5 of Scott Meyers, Effective Modern C++  

// doesn't work, error: dynamic initialization not supported for __device__  
//using funcpf = std::function<float(float,float)>;

//  error: dynamic initialization is not supported for __device__, __constant__ and __shared__ variables.
//__device__ funcpf funcfunc_d[2] = { maximum, minimum };

/**
 * @ref https://stackoverflow.com/questions/16498969/how-do-i-typedef-a-function-pointer-with-the-c11-using-syntax
 * How do I typedef a function pointer with the C++11 using syntax?  answer from 0x499602D2
 * */
using pf1 = std::add_pointer<int(int)>::type;  

__device__ int increment_k(int t) {
	return t+ 1;
}

__device__ int decrement_k(int t) {
	return t-1;
}

/**
 * @fn incdecops
 * @ref Stanley B. Lippman, Josee Lajoie, Barbara E. Moo. C++ Primer 
 * */
//  error: dynamic initialization is not supported for __device__, __constant__ and __shared__ variables.
/* __device__ std::map<int, pf1> incdecops = {
	{0, increment_k}, 
	{1, decrement_k} };
*/

__device__ pf1 d_func[2] = { increment_k, decrement_k };

__global__ void test_k(int *testArray, int funcId, uint numPoints ) {
	unsigned int index = threadIdx.x + __umul24(blockIdx.x,blockDim.x); 
	
	if (index >= numPoints) { return ; }
	
	pf1 func = d_func[funcId];

	// i=kx, kx + Nx*Mx, kx + 2*Nx*Mx, ... < n 
	// the big idea is to make sure all values of x=x[i], i=0,1,...n-1, gets calculated
	// this is because we worry about the case when n > Nx*Mx (or perhaps n >> Nx*Mx !!!)
	int Mx = blockDim.x; 
	int Nx = gridDim.x; 
	for (int idx = index; idx < numPoints; idx += Nx*Mx) {
		int temp = func(testArray[idx]);
		testArray[idx] = temp;
	}
		
}

void test(int *testArray, int funcId, unsigned int numPoints, int numThreadsPerBlock) {
	// define grid and block size
	unsigned int numBlocks = (numPoints + numThreadsPerBlock -1)/numThreadsPerBlock;
//	computeGridSize(numPoints, 256, numBlocks, numThreadsPerBlock);

	test_k<<<numBlocks, numThreadsPerBlock>>>( testArray, funcId, numPoints); 
}

int main (void)
{
	// sanity check
	constexpr const int Mx = THREAD_COUNT;
	std::cout << " Mx >> 1 : " << ( Mx >> 1) << std::endl;
	for (int test_i = Mx >> 1; test_i >0; test_i >>= 1) { std::cout << test_i << " " ; } std::cout << std::endl;

	constexpr const int ELEM_COUNT = 8 ;
    float x[ELEM_COUNT] = {-1.3f, 2.4f, 3.5f, -2.3f, 4.5f, 0.4f, -5.3f, -1.6f};

    float minimum = findExtremum(x, ELEM_COUNT, 1);
    float maximum = findExtremum(x, ELEM_COUNT, 0);

    printf("min=% 13.6e  max=% 13.6e\n", minimum, maximum);

	std::cout << std::endl << "increment, decrement " << std::endl ;

	int numPoints = 10;
	
	// on host
	std::vector<int> testArray(numPoints,0);
	for (uint i=0; i<numPoints;i++) { testArray[i] = i; }
	
	auto deleterZZ_lambda=[&](int* ptr){ cudaFree(ptr); };
	std::unique_ptr<int[], decltype(deleterZZ_lambda)> d_test(new int[numPoints], deleterZZ_lambda); // u for unique and unified
	cudaMallocManaged((void **) &d_test, numPoints*sizeof(float));

	std::cout << " Before : " << std::endl;
	for (auto ele : testArray) { std::cout << ele << " " ; } std::cout << std::endl;

	cudaMemcpy(d_test.get(),testArray.data(),numPoints*sizeof(int),cudaMemcpyHostToDevice); 
	
	test(d_test.get(), 0, numPoints, 16);
	
	cudaMemcpy(testArray.data(),d_test.get(), numPoints*sizeof(int),cudaMemcpyDeviceToHost); 
	 
	std::cout << " After : " << std::endl; 
	for (auto ele : testArray) { std::cout << ele << " " ; } std::cout << std::endl;


    return EXIT_SUCCESS;
}

