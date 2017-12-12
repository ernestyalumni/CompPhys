/**
 * 	@file 	helloworld.cu
 * 	@brief 	Following example shows a simple Hello World program incorporating dynamic parallelism.  
 * 	@ref	http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-guidelines
 * 	@details 
 * 	 
 * COMPILATION TIP :  nvcc -arch=sm_35 -rdc=true hello_world.cu -o hello -lcudadevrt
 * 
 * */
#include <stdio.h>

__global__ void childKernel() 
{
	printf("Hello ");
}

__global__ void parentKernel()
{
	// launch child
	childKernel<<<1,1>>>();
	if (cudaSuccess != cudaGetLastError()) {
		return;
	}
	
	// wait for child to complete
	if (cudaSuccess != cudaDeviceSynchronize()) {
		return; 
	}
	
	printf("World!\n");
}

int main(int argc, char *argv[]) 
{
	// launch parent
	parentKernel<<<1,1>>>();
	if (cudaSuccess != cudaGetLastError()) {
		return 1;
	}
	
	// wait for parent to complete
	if (cudaSuccess != cudaDeviceSynchronize()) {
		return 2;
	}
	
	return 0;
}
