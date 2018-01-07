/**
 * @file   : cg_eg.cu
 * @brief  : Cooperative Groups (cg) examples, in CUDA C/C++11  
 * @author : Ernest Yeung <ernestyalumni@gmail.com>
 * @date   : 20180106
 * @ref    :  
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
 * nvcc -std=c++11 smart_ptrs_arith.cu -o smart_ptrs_arith.exe
 * 
 * */
#include <cooperative_groups.h>   
#include <algorithm> // std::fill_n
#include <iostream>  
 
namespace cg = cooperative_groups; 

__device__ int reduce_sum(cg::thread_group g, int *temp, int val) 
{
	int lane = g.thread_rank(); 
	
	// Each iteration halves the number of active threads 
	// Each thread adds its partial sum[i] to sum[lane+i]
	for (int i = g.size() / 2; i > 0; i/= 2) 
	{
		temp[lane] = val; 
		g.sync(); // wait for all threads to store 
		if (lane<i) { 
			val += temp[lane+i]; 
		}
		g.sync(); 	// wait for all threads to load 
	}
	return val; // note: only thread 0 will return full sum
};  

/** @fn thread_sum
 * @brief compute many partial sums in parallel, 
 * @details compute many partial sums in parallel, 
 * 	where each thread strides through the array computing a partial sum
 * */

__device__ int thread_sum(int *input, int n) 
{
	int sum = 0;
	
	for (int i = threadIdx.x + blockIdx.x * blockDim.x; 
		i < n/ 4;
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
//	auto g = cg::this_thread_block(); 
	cg::thread_block g = cg::this_thread_block();
	int block_sum = reduce_sum(g, temp, my_sum); 
		
	if (g.thread_rank() == 0) { 
		atomicAdd(sum, block_sum); 
	}
};  

int main(int argc, char* argv[] ) {
	int n = 1<<24; 
	int blockSize = 256; 
	int nBlocks = (n + blockSize - 1)/ blockSize; 
	int sharedBytes = blockSize * sizeof(int);  
	
	int *sum, *data; 
	cudaMallocManaged(&sum, sizeof(int)); 
	cudaMallocManaged(&data, n * sizeof(int)); 
	std::fill_n(data,n,1); // initialize data
	cudaMemset(sum, 0, sizeof(int)); 
	
	sum_kernel_block<<<nBlocks, blockSize, sharedBytes>>>(sum,data,n);  

	int h_sum;
	cudaMemcpy(&h_sum, sum, sizeof(int),cudaMemcpyDeviceToHost); 
	std::cout << " h_sum : " << h_sum << std::endl;	
	
	cudaFree(data);  
		
	/* try other values */  
	n = 1<<25; 
	nBlocks = (n + blockSize - 1)/blockSize; 
	cudaMallocManaged(&data, n * sizeof(int)); 
	std::fill_n(data,n,1); // initialize data
	cudaMemset(sum, 0, sizeof(int)); 
	
	sum_kernel_block<<<nBlocks, blockSize, sharedBytes>>>(sum,data,n);  

	/* sanity check; read out values on host */
	cudaMemcpy(&h_sum, sum, sizeof(int),cudaMemcpyDeviceToHost); 
	std::cout << " h_sum : " << h_sum << std::endl;	
	cudaFree(data);  

	/* try other values */  
	n = 1<<30; 
	nBlocks = (n + blockSize - 1)/blockSize; 
	cudaMallocManaged(&data, n * sizeof(int)); 
	std::fill_n(data,n,1); // initialize data
	cudaMemset(sum, 0, sizeof(int)); 
	
	sum_kernel_block<<<nBlocks, blockSize, sharedBytes>>>(sum,data,n);  

	/* sanity check; read out values on host */
	cudaMemcpy(&h_sum, sum, sizeof(int),cudaMemcpyDeviceToHost); 
	std::cout << " h_sum : " << h_sum << std::endl;	
	cudaFree(data);  
		
	/* try other values. n = 1<<30, 2^(30), seems to be the "limit" */  
	n = 1<<31; 
	/* sanity check of what n is as an int on host */
	std::cout << std::endl << " n = 1<<31 : " << n << std::endl << std::endl; 
/*	nBlocks = (n + blockSize - 1)/blockSize; 
	cudaMallocManaged(&data, n * sizeof(int)); 
	std::fill_n(data,n,1); // initialize data
	cudaMemset(sum, 0, sizeof(int)); 
	
	sum_kernel_block<<<nBlocks, blockSize, sharedBytes>>>(sum,data,n);  

	// sanity check; read out values on host 
	cudaMemcpy(&h_sum, sum, sizeof(int),cudaMemcpyDeviceToHost); 
	std::cout << " h_sum : " << h_sum << std::endl;	
	cudaFree(data);  
*/
		
	cudaFree(sum);	
}
	

	
	

