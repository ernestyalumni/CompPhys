/**
 * @file   : cg_eg.cu
 * @brief  : Examples of using cooperative groups
 * @details : cooperative groups for CUDA examples
 *  
 * @author : Ernest Yeung <ernestyalumni@gmail.com>
 * @date   : 20170104      
 * @ref    : https://devblogs.nvidia.com/parallelforall/cooperative-groups/
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
 * nvcc cg_eg.cu -o cg_eg
 * 
 * */
#include <cooperative_groups.h>  
#include <stdio.h>  
#include <iostream>
#include <algorithm>  // std::fill_n

namespace cg = cooperative_groups;



/** @fn explore_t_blocks
 * @brief explore what thread blocks are with cg 
 * */
template <unsigned int blockSize>
__global__ void explore_t_blocks()
{
	// unsized shared memory arrays
	extern __shared__ int _smem[];  


	// Handle to thread block group  
	auto cta = cg::this_thread_block(); 
	// also works 
//	cg::thread_block cta = cg::this_thread_block(); 
	
	
	unsigned int tid = threadIdx.x; 
	unsigned int i = blockIdx.x * blockSize*2 + threadIdx.x;  
	unsigned int gridSize = blockSize*2*gridDim.x; 
	unsigned int k_x = threadIdx.x + blockIdx.x * blockDim.x; 
	
	printf("tid: %d  blockIdx.x: %d  thread_block->size: %d  ->thread_rank: %d   \n", tid, blockIdx.x, cta.size(),cta.thread_rank()); 

//	cta.is_valid(); // error: class "cooperative_groups::__v1::thread_block" has no member "is_valid"

	dim3 cgi = cta.group_index();
	dim3 cti = cta.thread_index(); 
	printf("group_index: gi.x: %d  gi.y: %d  gi.z: %d \n", cgi.x,cgi.y,cgi.z); 
	printf("thread_index: ti.x: %d  ti.y: %d  ti.z: %d\n", cti.x,cti.y,cti.z); 
		
};

__device__ int reduce_sum(cg::thread_group g, int *temp, int val) 
{
	int lane = g.thread_rank(); 
	
	// Each iteration halves number of active threads 
	// Each thread adds its partial sum[i] to sum[lane+i]
	for (int i=g.size() / 2; i > 0; i /= 2) 
	{
		temp[lane] = val; 
		g.sync(); // wait for all threads to store 
		if (lane <i) { 
			val += temp[lane+i]; 
		}
		g.sync(); 	// wait for all threads to load 
	}	
	return val; // note: only thread 0 will return full sum  
};

/*
__device__ unsigned long long int reduce_sum(cg::thread_group g, unsigned long long int *temp, unsigned long long int val) 
{
	int lane = g.thread_rank(); 
	
	// Each iteration halves number of active threads 
	// Each thread adds its partial sum[i] to sum[lane+i]
	for (int i=g.size() / 2; i > 0; i /= 2) 
	{
		temp[lane] = val; 
		g.sync(); // wait for all threads to store 
		if (lane <i) { 
			val += temp[lane+i]; 
		}
		g.sync(); 	// wait for all threads to load 
	}	
	return val; // note: only thread 0 will return full sum  
}
*/

__device__ int thread_sum(int *input, int n) 
{
	int sum = 0; 
	
	for (int i = threadIdx.x + blockIdx.x * blockDim.x ; 
			i < n/4;
			i += blockDim.x * gridDim.x) 
	{ 
		int4 in = ((int4*) input)[i]; 
		sum += in.x + in.y + in.z + in.w; 
	}
	return sum; 
};



__global__ void sum_kernel_block(int *sum, int *input, int n)
{
	int my_sum = thread_sum(input, n);
	
	extern __shared__ int temp[]; 
	auto g = cg::this_thread_block(); 
	int block_sum = reduce_sum(g, temp, my_sum); 
	
	if (g.thread_rank() ==0) {
		atomicAdd(sum, block_sum); 
	}
};

/*
__global__ void sum_kernel_block(unsigned long long int *sum, int *input, int n)
{
	unsigned long long int my_sum = thread_sum(input, n);
	
	extern __shared__ unsigned long long int temp[]; 
	auto g = cg::this_thread_block(); 
	unsigned long long int block_sum = reduce_sum(g, temp, my_sum); 
	
	if (g.thread_rank() ==0) {
		atomicAdd(sum, my_sum); 
	}
}
*/


int main(int argc, char* argv[]) {
	explore_t_blocks<32><<<2,4, 2*sizeof(int)>>>();
	cudaDeviceSynchronize(); // https://stackoverflow.com/questions/15669841/cuda-hello-world-printf-not-working-even-with-arch-sm-20

	std::cout << std::endl << std::endl; 
	explore_t_blocks<32><<<4,8, 2*sizeof(int)>>>();
	cudaDeviceSynchronize(); // https://stackoverflow.com/questions/15669841/cuda-hello-world-printf-not-working-even-with-arch-sm-20

//	cudaDeviceReset();
	
	/* ******************************
	 * driver commands/function calls to test the computation of the sum of a 16M-element array 
	 * ****************************** */  
	
	
	int n = 1<<24; 

	std::cout << std::endl << std::endl << " int n = 1 << 24 : " << n << std::endl << std::endl; 
	
	/* ************************* */
	/* thread block, grid dims.  */
	/* ************************* */
	int blockSize = 256; 
	int nBlocks = (n + blockSize - 1) / blockSize; 
	int sharedBytes = blockSize * sizeof(int); 
//	int sharedBytes = blockSize * sizeof(unsigned long long int); 
	
	
	int *sum, *data; 
//	unsigned long long int *sum;
//	int *data; 
	cudaMallocManaged(&sum, sizeof(int)) ;
	cudaMallocManaged(&data, n * sizeof(int)) ;
//	cudaMallocManaged(&sum, sizeof(unsigned long long int)) ;
//	cudaMallocManaged(&data, n * sizeof(int)) ;
	std::fill_n(data, n , 1); // initialize data
	cudaMemset(sum, 0, sizeof(int)) ; 
//	cudaMemset(sum, 0, sizeof(unsigned long long int)) ; 
		
	sum_kernel_block<<<nBlocks, blockSize, sharedBytes>>>(sum, data, n);  	
//	cudaDeviceSynchronize();
	

	/* sanity check; read result of summation in host */
	int h_sum; 
//	unsigned long long int h_sum;
	//cudaMemcpy( &h_sum, sum, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);  
	cudaMemcpy( &h_sum, sum, sizeof(int), cudaMemcpyDeviceToHost);  

	std::cout << std::endl << std::endl << " h_sum : " << h_sum << std::endl;  

	cudaFree(data); 
	cudaFree(sum);

}
