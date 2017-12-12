/**
 * 	@file 	helloworld1.cu
 * 	@brief 	Following example shows a simple Hello World program incorporating dynamic parallelism.  
 * 	@ref	http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-guidelines
 * 	@details When changing childKernel<<<1,1>>>();  
 * 	childKernel<<<1,2>>>(); //  Hello  Hello World!  
 * 	childKernel<<<1,3>>>(); //  Hello  Hello  Hello World! 
 *  childKernel<<<2,1>>>(); //  Hello  Hello World! 
 * 	childKernel<<<3,1>>>(); //  Hello  Hello  Hello World! 
 * 	childKernel<<<2,2>>>(); //  Hello  Hello  Hello  Hello World! 
 * 	fix childKernel<<<1,1>>>(); change parentKernel<<<1,1>>>();
 * 	parentKernel<<<2,1>>>(); //  Hello  Hello World! 
World! 
 * 		parentKernel<<<3,1>>>(); //  Hello  Hello  Hello World! 
World! 
World! 
 * 	parentKernel<<<1,2>>>(); //  Hello  Hello World! 
World! 
 *	parentKernel<<<1,3>>>(); //  Hello  Hello  Hello World! 
World! 
World!  
 * 	parentKernel<<<2,2>>>(); //  Hello  Hello  Hello  Hello World! 
World! 
World! 
World! 
 * 
 * COMPILATION TIP :  nvcc -arch=sm_35 -rdc=true hello_world.cu -o hello -lcudadevrt
 * nvcc -arch=sm_52 -rdc=true helloworld1.cu -o helloworld1 
 * removing -arch=sm_52 results in this error 
 * error: calling a __global__ function("childKernel") from a __global__ function("parentKernel") 
 * is only allowed on the compute_35 architecture or above
 * */
#include <stdio.h>
#include <iostream> // std::cout 

__global__ void childKernel()
{
//	std::cout << " Hello " ; // error: identifier "std::cout" is undefined in device code
	printf(" Hello ");
}

__global__ void parentKernel() 
{
	// launch child
	childKernel<<<1,1>>>();
	
	// wait for child to complete
	if (cudaSuccess != cudaDeviceSynchronize()) {
		return; 
	} 
	
//	std::cout << "World! " << std::endl; // error: identifier "std::cout" is undefined in device code
	printf("World! \n");
}

int main(int argc, char *argv[]) 
{
	// launch parent
	parentKernel<<<2,2>>>();
	if (cudaSuccess != cudaGetLastError()) {
		std::cout << " returning 1 for cudaSuccess != cudaGetLastError() " << std::endl; 
		return 1;
	}
	
	// wait for parent to complete
	if (cudaSuccess != cudaDeviceSynchronize()) {
		std::cout << " returning 2 for cudaSuccess != cudaDeviceSynchronize() "<< std::endl; 
		return 2; 
	}
	
	return 0;
}
	
