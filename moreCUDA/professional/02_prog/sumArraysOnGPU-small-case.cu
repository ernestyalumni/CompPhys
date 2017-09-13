/**
 * @file   : simpleDivergence_b.cu
 * @brief  : Simple warp divergence, with my editting
 * @author : Ernest Yeung <ernestyalumni@gmail.com>
 * @date   : 20170910  
 * @ref    :  https://github.com/deeperlearning/professional-cuda-c-programming/blob/master/examples/chapter03/simpleDivergence.cu
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
 * COMPILATION TIP
 * nvcc --relocatable-device-code true simpleDivergence.cu -o simpleDivergence.exe
 * 
 * */
#include <cuda_runtime.h>
#include <stdio.h>

#define CHECK(call) 														\
{																			\
	const cudaError_t error = call;											\
	if (error != cudaSuccess)												\
	{																		\
		printf("Error: %s:%d, ", __FILE__, __LINE__);						\
		printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));	\
		exit(1);															\
	}																		\
}																			\

void checkResult(float *hostRef, float *gpuRef, const int N) {
	double epsilon = 1.0E-8;
	bool match = 1;
	for (int i=0; i<N; i++) {
		if (abs(hostRef[i] - gpuRef[i]) > epsilon) {
			match = 0;
			printf("Arrays do not match!\n");
			printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i], gpuRef[i], i);
			break;
		}
	}
	
	if (match) printf("Arrays match.\n\n");
}

void initialData(float *ip, int size) {
	// generate different seed for random number
	time_t t;
	srand((unsigned) time(&t));
	
	for (int i=0; i<size; i++) {
		ip[i] = (float)( rand() & 0xFF )/10.0f;
	}
}

void sumArraysOnHost(float *A, float *B, float *C, const int N) {
	for (int idx=0; idx<N; idx++) 
		C[idx] = A[idx] + B[idx];
}

__global__ void sumArraysOnGPU(float *A, float *B, float *C) {
	int i = threadIdx.x;
	C[i] = A[i] + B[i]; 
}

int main(int argc, char **argv) {
	printf("%s Starting ... \n", argv[0]);
	
	// set up device
	int dev = 0;
	cudaSetDevice(dev);
	
	// set updata size of vectors
	int nElem = 32;
	printf("Vector size %d\n", nElem);
	
	// malloc host memory
	size_t nBytes = nElem * sizeof(float);
	
	float *h_A, *h_B, *hostRef, *gpuRef;
	h_A		= (float *)malloc(nBytes);
	h_B 	= (float *)malloc(nBytes);
	hostRef = (float *)malloc(nBytes);
	gpuRef  = (float *)malloc(nBytes);
	
	// initialize data at host side
	initialData(h_A, nElem);
	initialData(h_B, nElem);
	
	memset(hostRef, 0, nBytes);
	memset(gpuRef, 0, nBytes);
	
	// malloc device global memory
	float *d_A, *d_B, *d_C;
	cudaMalloc((float **)&d_A, nBytes);
	cudaMalloc((float **)&d_B, nBytes);
	cudaMalloc((float **)&d_C, nBytes);
	
	// transfer data from host to device
	cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);
	
	// invoke kernel at host side
	dim3 block (nElem);
	dim3 grid (nElem/block.x);
	
	sumArraysOnGPU<<< grid, block >>>(d_A, d_B, d_C);
	printf("Execution configuration <<<%d, %d>>>\n", grid.x,block.x);
	
	// copy kernel result back to host side
	cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
	
	// add vector at host side for result checks
	sumArraysOnHost(h_A, h_B, hostRef, nElem);
	
	// check device results
	checkResult(hostRef, gpuRef, nElem);
	
	// free device global memory
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	
	// free host memory
	free(h_A);
	free(h_B);
	free(hostRef);
	free(gpuRef);
	
	return(0);
}
	
	
