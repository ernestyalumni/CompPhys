/**
 * 	@file 	parentchildcoherence.cu
 * 	@brief 	Following example shows parent and child grids have coherent access to global memory, 
 * 	with weak consistency guarantees between child and parent.    
 * 	@ref	http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#global-memory
 * 	@details n the following example, the child grid executing child_launch is 
 * 	only guaranteed to see the modifications to data made before the child grid was launched. 
 * 	Since thread 0 of the parent is performing the launch, 
 * 	the child will be consistent with the memory seen by thread 0 of the parent. 
 * 	Due to the first __syncthreads() call, the child will see data[0]=0, data[1]=1, ..., data[255]=255 
 * 	(without the __syncthreads() call, only data[0] would be guaranteed to be seen by the child). 
 * 	When the child grid returns, thread 0 is guaranteed to see modifications made by the threads in its child grid. 
 * 	Those modifications become available to the other threads of the parent grid 
 * 	only after the second __syncthreads() call:
 * 	 
 * COMPILATION TIP :  nvcc -arch=sm_52 -rdc=true parentchildcoherence.cu -o parentchildcoherence 
 * nvcc -arch=sm_52 -rdc=true parentchildcoherence.cu -o parentchildcoherence.exe
 * 
 * */
#include <stdio.h> 
#include <iostream> // std::cout 

__global__ void child_launch(int *data) {
	data[threadIdx.x] = data[threadIdx.x] + 1;
}

__global__ void parent_launch(int *data) {
	data[threadIdx.x] = threadIdx.x;
	
	__syncthreads(); 
	
	if (threadIdx.x == 0) {
		child_launch<<<1,256>>>(data);
		cudaDeviceSynchronize(); 
	}
	
	__syncthreads();
}

void host_launch(int *data) {
	parent_launch<<<1, 256>>>(data); 
}

int main(int argc, char *argv[]) 
{
	constexpr const int L = 256; // L is the size or length of the int array
	
	// initialize and assign values of host int array (on the CPU RAM stack?)  
	int data[L]; 
	for (int idx=0; idx< L; idx++) {
//		data[idx] = idx;
		data[idx] = 0;
	}

	// sanity check, print out host values
	for (int idx=0; idx< L; idx++) {
		std::cout << data[idx] << " " ; 
	} std::cout << std::endl << std::endl; 
	
	
	// initialize device int array 
//	int d_data[L]; 
	int * d_data;
	cudaMallocManaged((void **) &d_data, L * sizeof(int)); 
	
	// copy host initialized values to device
	cudaMemcpy( d_data, data, L * sizeof(int), cudaMemcpyHostToDevice); 
	
	host_launch( d_data); 
	
	cudaDeviceSynchronize();
	
	// copy back values on device, after host_launch, to host
	cudaMemcpy( data, d_data, L * sizeof(int), cudaMemcpyDeviceToHost); 

	// sanity check, print out host values
	std::cout << std::endl;
	for (int idx=0; idx< L; idx++) {
		std::cout << data[idx] << " " ; 
	} std::cout << std::endl << std::endl; 

	
	cudaFree(d_data);
} 

