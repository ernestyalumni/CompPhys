/**
 * @file   : params_kernelf_og.cu
 * @brief  : Original implementation from njuffa, verbotim; 
 * 				CUDA kernel functions as parameters with CUDA C++14, CUDA Unified Memory Management
 * @details : Original implementation from njuffa, verbotim 
 * 				std::function vs. function pointer in C++11, C++14, and now in CUDA
 * 				std::function vs. function pointer for CUDA kernel functions (i.e. __global__ )
 * 
 * @author : Ernest Yeung <ernestyalumni@gmail.com>
 * @date   : 20171020  
 * @ref    : https://devtalk.nvidia.com/default/topic/487190/kernel-functions-as-parameters-/
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
 * nvcc -std=c++14 params_kernelf_og.cu -o params_kernelf_og.exe
 * 
 * */
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_COUNT  240
#define THREAD_COUNT 128

// pf = processing function
// C style
typedef float (*pf)(float a, float b);

/** @brief function pointer example, pf = "processing function"
 * @ref Scott Meyers, Effective Modern C++, pp. 63, Item 9
 * */
//using pf = float (*)(float, float);

__device__ float minimum(float a, float b)
{
    return fminf(a,b);
};

__device__ float maximum(float a, float b)
{
    return fmaxf(a,b);
};

/**
 * @brief 
 * @ref Scott Meyers, Effective Modern C++, pp. 63, Item 9
*/
__device__ pf func_d[2] = { maximum, minimum };

__shared__ float partExtr[THREAD_COUNT];

__device__ void minmax(float *x, float *res, int n, pf func)
{
//	__shared__ float partExtr[THREAD_COUNT];
    int i;
    int tid = threadIdx.x;

    float extr = x[0];

    for (i = THREAD_COUNT*blockIdx.x+tid; i < n; i += gridDim.x*THREAD_COUNT) {
        extr = func (extr, x[i]);
    }

    partExtr[tid] = extr;

    for (i = THREAD_COUNT >> 1; i > 0; i >>= 1) {
        __syncthreads(); 

        if (tid < i) {
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


float findExtremum (float *x, int n, int findmin)
{
    pf func_h[2] = { fmaxf, fminf };

    float *res_d;
    float *res_h;
    float *x_d;
    float r;

    if (n < 1) return sqrtf(-1.0f); // NaN

    cudaMalloc((void**)&res_d, BLOCK_COUNT*sizeof(res_d[0]));
    cudaMalloc((void**)&x_d, n * sizeof(x_d[0]));

    cudaMemcpy(x_d, x, n * sizeof(x_d[0]), cudaMemcpyHostToDevice);

    minmax_kernel<<<BLOCK_COUNT,THREAD_COUNT>>>(x_d, res_d, n, !!findmin);

    res_h = (float *)malloc (BLOCK_COUNT * sizeof(res_h[0]));

    if (!res_h) {
		fprintf(stderr, "res_h allocation failed\n");
		exit(EXIT_FAILURE);
    }

    cudaMemcpy(res_h, res_d, BLOCK_COUNT * sizeof(res_d[0]), cudaMemcpyDeviceToHost);

    cudaFree(res_d);
    cudaFree(x_d);

    r = res_h[0];

    for (int i = 1; i < BLOCK_COUNT; i++) {
		r = func_h[findmin](r, res_h[i]);
	}

    free(res_h);

    return r;
}




int main (void)
{
	constexpr const int ELEM_COUNT = 8 ;
    float x[ELEM_COUNT] = {-1.3f, 2.4f, 3.5f, -2.3f, 4.5f, 0.4f, -5.3f, -1.6f};

    float minimum = findExtremum(x, ELEM_COUNT, 1);

    float maximum = findExtremum(x, ELEM_COUNT, 0);

    printf("min=% 13.6e  max=% 13.6e\n", minimum, maximum);

    return EXIT_SUCCESS;
}





/**
 * 	@brief std::function 
 * 	@details "The type of a std::function-declared variable holding a closure 
 * 				is an instantiation of std::function template, and that has 
 * 				fixed size for any given signature.  
 * 				This size may not be adequate for the closure it's asked to store, 
 * 				and when that's the case, std::function constructor will allocate
 * 				heap memory to store the closure.  
 * @ref Scott Meyers, Effective Modern C++, pp. 39 Item 5 
 * */
