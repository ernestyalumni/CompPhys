/**
 * @file   : cg_eg3.cu
 * @brief  : Examples of using cooperative groups
 * @details : cooperative groups for CUDA examples
 *  Note; limitations on maximum values that can be reduced (summation) is due to 32-bit architecture of 
 * GeForce GTX 980 Ti that I'm using; please make a hardware donation (for a Titan V or GTX 1080 Ti) if you find this code useful!  
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
 * nvcc cg_eg3.cu -o cg_eg3
 * 
 * */
#include <cooperative_groups.h>  
#include <iostream>
#include <algorithm>  // std::fill_n
#include <memory> // std::unique_ptr

/* ********** functions to setup device GPU, test values ********** */

/** @fn getMaxGridSize
 * @brief get maxGridSize (total number threads on a (thread) grid, on device GPU, of a single device GPU
 * */
size_t get_maxGridSize() {
	cudaDeviceProp prop;
	int count;
	cudaGetDeviceCount(&count);
	size_t MAXGRIDSIZE; 
	if (count>0) {
		cudaGetDeviceProperties(&prop, 0);
		MAXGRIDSIZE = prop.maxGridSize[0]; 
		return MAXGRIDSIZE; 
	} else { return EXIT_FAILURE; }
}; 

__global__ void inc_kernel(int *input,int inc, int L) {
	unsigned int k_x = threadIdx.x + blockDim.x*blockIdx.x; 
	for (unsigned int idx=k_x; idx < L; idx += blockDim.x*gridDim.x) {
		input[idx] = ((int) idx + inc); 
	}
}

/* ********** END of functions to setup device GPU, test values ********** */

namespace cg = cooperative_groups;  

/** @fn reduce_sum
 * @details used to reduce (summation) on a single thread block in shared memory 
 * while not obvious from this function definition, in practical usage, 
 * val will be the partial sum that is at the index given by the global thread index 
 * threadIdx.x + blockDim.x * blockIdx.x; 
 * and so we'll have loaded all the various array values for this particular thread block into 
 * shared memory lane
 * */
__device__ int reduce_sum(cg::thread_group g, int *temp, int val) 
{
	int lane = g.thread_rank(); 
	
	// Each iteration halves the number of active threads
	// Each thread adds to partial sum[i] its sum[lane+i] 
	for (int i = g.size() / 2; i >0; i/=2) 
	{
		// load the array values with this thread block into temp
		temp[lane] = val;
		g.sync(); 	// wait for all threads to store 
		if (lane <i) {
			val += temp[lane+i]; 
		}
		g.sync(); 		// wait for all threads to load 
	}
	return val; 	// note: only thread 0 will return full sum
}; 

/** @fn thread_sum
 * @brief compute many partial sums in parallel, Generalized to when n is not a power of 2
 * @details compute many partial sums in parallel, Generalized to when n is not a power of, 
 * 	where each thread strides through the array computing a partial sum 
 * */ 
__device__ int thread_sum(int *input, int L) 
{
	int sum =0;

	unsigned int k_x = threadIdx.x + blockDim.x*blockIdx.x; 
	
	/* increment by blockDim.x*gridDim.x, so that a single thread will do all the 
	 * "work" needed done on n, especially if n >= gridDim.x*blockDim.x = N_x*M_x */	
	for (int i=k_x; 
			i < L/4; 
			i += blockDim.x * gridDim.x) 
	{
		int4 in = ((int4*) input)[i]; 
		sum += in.x + in.y + in.z + in.w; 
	}
	// process remaining elements
	for (unsigned int idx= k_x + L/4*4; idx < L; idx += 4 ) {
		sum += input[idx];
	}
	
	return sum; 
};  

/** @fn sum_kernel
 * @brief sum kernel, generalized for n not a power of 2 */
__global__ void sum_kernel(int *sum, int *input, int L)
{
	// for a particular thread k_x, we've obtained the 
	// sum of input[k_x], input[k_x+1], ... input[k_x+3] in sum4 
	int sum4 = thread_sum(input, L); 
	
	extern __shared__ int temp[];
	auto g = cg::this_thread_block();
	int block_sum = reduce_sum(g,temp,sum4); 
	
	if (g.thread_rank() == 0) {
		atomicAdd(sum, block_sum); 
	}
};  

int main(int argc, char* argv[]) 
{
	size_t MAXGRIDSIZE = get_maxGridSize();  
	/* ***** (thread) grid,block dims ***** */ 
	/* min of N_x, number of (thread) blocks on grid in x-direction, and MAX_BLOCKS allowed is 
	 * determined here */
	unsigned int M_x = 1<<6;  // M_x = number of threads in x-direction, in a single block, i.e. blocksize; 2^8=256  
	unsigned int L = 1<<7; // doesn't output correct values for n = 1<<30    
	unsigned int MAX_BLOCKS = (MAXGRIDSIZE + M_x - 1)/ M_x; 
	// notice how we're only launching 1/4 of L threads
	unsigned int N_x = min( MAX_BLOCKS, ((L/4 + M_x - 1)/ M_x)); 
	int sharedBytes = M_x * sizeof(int);  
	/* ***** END of (thread) grid,block dims ***** */ 

	// setup input, output 
	auto del_ints_lambda=[&](int* ptr) { cudaFree(ptr); }; 
	std::unique_ptr<int,decltype(del_ints_lambda)> sum(nullptr,del_ints_lambda); 
	std::unique_ptr<int[],decltype(del_ints_lambda)> input(nullptr,del_ints_lambda); 
	cudaMallocManaged((void**)&sum, sizeof(int)) ;
	cudaMallocManaged((void**)&input, L*sizeof(int));
	std::fill_n(input.get(),L,1); 
	cudaMemset(sum.get(), 0,sizeof(int));

	sum_kernel<<<N_x,M_x,sharedBytes>>>(sum.get(),input.get(),L); 

	/* sanity check */
	// host output of sum 
	std::unique_ptr<int> h_sum = std::make_unique<int>( 0 );  
	cudaMemcpy( h_sum.get(), sum.get(), 1*sizeof(int), cudaMemcpyDeviceToHost);  
	std::cout << std::endl << " *h_sum : " << *h_sum << " 1<<7 : " << (1<<7) << std::endl; 

	/* ******************************************************* */
	/* ********** more tests of \sum_{i=1}^L 1 = L ********** */
	/* ***** L = 1<<8 = 2^8 = 256 test ***** */
	L = 1<< 8; 
	// notice how we're only launching 1/4 of L threads
	N_x = min( MAX_BLOCKS, ((L/4 + M_x - 1)/ M_x)); 

	std::unique_ptr<int[],decltype(del_ints_lambda)> input1(nullptr,del_ints_lambda); 
	cudaMallocManaged((void**) &input1,L*sizeof(int)); 
	std::fill_n(input1.get(),L,1); 
	cudaMemset(sum.get(), 0,sizeof(int));
	
	sum_kernel<<<N_x,M_x,sharedBytes>>>(sum.get(),input1.get(),L); 
	
	/* sanity check */
	// host output of sum 
	cudaMemcpy( h_sum.get(), sum.get(), 1*sizeof(int), cudaMemcpyDeviceToHost);  
	std::cout << std::endl << " *h_sum : " << *h_sum << " 1<<8 : " << (1<<8) << std::endl; 

	/* ***** L = 1<<9 + 1= 2^9 + 1= 513 test ***** */ 
	L = (1<< 9)+1; 
	// notice how we're only launching 1/4 of L threads
	N_x = min( MAX_BLOCKS, ((L/4 + M_x - 1)/ M_x)); 

	std::unique_ptr<int[],decltype(del_ints_lambda)> input2(nullptr,del_ints_lambda); 
	cudaMallocManaged((void**) &input2,L*sizeof(int)); 
	std::fill_n(input2.get(),L,1); 
	cudaMemset(sum.get(), 0,sizeof(int));
	
	sum_kernel<<<N_x,M_x,sharedBytes>>>(sum.get(),input2.get(),L); 
	
	/* sanity check */
	// host output of sum 
	cudaMemcpy( h_sum.get(), sum.get(), 1*sizeof(int), cudaMemcpyDeviceToHost);  
	std::cout << std::endl << " *h_sum : " << *h_sum << " (1<<9) + 1 : " << ((1<<9)+1) << std::endl; 

	/* ***** L = 1<<29 = 2^29 test ***** */ 
	{
	L = (1<< 29); 
	// notice how we're only launching 1/4 of L threads
	N_x = min( MAX_BLOCKS, ((L/4 + M_x - 1)/ M_x)); 

	std::unique_ptr<int[],decltype(del_ints_lambda)> input3(nullptr,del_ints_lambda); 
	cudaMallocManaged((void**) &input3,L*sizeof(int)); 
	std::fill_n(input3.get(),L,1); 
	cudaMemset(sum.get(), 0,sizeof(int)); // reset the sum 
	
	sum_kernel<<<N_x,M_x,sharedBytes>>>(sum.get(),input3.get(),L); 
	
	/* sanity check */
	// host output of sum 
	cudaMemcpy( h_sum.get(), sum.get(), 1*sizeof(int), cudaMemcpyDeviceToHost);  
	std::cout << std::endl << " *h_sum : " << *h_sum << " (1<<29) : " << (1<<29) << std::endl; 
	}

	/* ***** L = (1<<29) + 2 = (2^29 + 2) test ***** */ 
	{
	L = (1<< 29)+2; 
	// notice how we're only launching 1/4 of L threads
	N_x = min( MAX_BLOCKS, ((L/4 + M_x - 1)/ M_x)); 

	std::unique_ptr<int[],decltype(del_ints_lambda)> input4(nullptr,del_ints_lambda); 
	cudaMallocManaged((void**) &input4,L*sizeof(int)); 
	std::fill_n(input4.get(),L,1); 
	cudaMemset(sum.get(), 0,sizeof(int)); // reset the sum 
	
	sum_kernel<<<N_x,M_x,sharedBytes>>>(sum.get(),input4.get(),L); 
	
	/* sanity check */
	// host output of sum 
	cudaMemcpy( h_sum.get(), sum.get(), 1*sizeof(int), cudaMemcpyDeviceToHost);  
	std::cout << std::endl << " *h_sum : " << *h_sum << " (1<<29)+2 : " << ((1<<29)+2) << std::endl; 
	}

	/* ***** L = 1<<30 = 2^30 test ***** */ 
	{
	L = (1<< 30); 
	// notice how we're only launching 1/4 of L threads
	N_x = min( MAX_BLOCKS, ((L/4 + M_x - 1)/ M_x)); 

	std::unique_ptr<int[],decltype(del_ints_lambda)> input4(nullptr,del_ints_lambda); 
	cudaMallocManaged((void**) &input4,L*sizeof(int)); 
	std::fill_n(input4.get(),L,1); 
	cudaMemset(sum.get(), 0,sizeof(int)); // reset the sum 
	
	sum_kernel<<<N_x,M_x,sharedBytes>>>(sum.get(),input4.get(),L); 
	
	/* sanity check */
	// host output of sum 
	cudaMemcpy( h_sum.get(), sum.get(), 1*sizeof(int), cudaMemcpyDeviceToHost);  
	std::cout << std::endl << " *h_sum : " << *h_sum << " (1<<30) : " << (1<<30) << std::endl; 
	}
	
	/* ***** L = 1<<30 +3 = 2^30+3 test ***** */ 
	{
	L = (1<< 30)+3; 
	// notice how we're only launching 1/4 of L threads
	N_x = min( MAX_BLOCKS, ((L/4 + M_x - 1)/ M_x)); 

	std::unique_ptr<int[],decltype(del_ints_lambda)> input4(nullptr,del_ints_lambda); 
	cudaMallocManaged((void**) &input4,L*sizeof(int)); 
	std::fill_n(input4.get(),L,1); 
	cudaMemset(sum.get(), 0,sizeof(int)); // reset the sum 
	
	sum_kernel<<<N_x,M_x,sharedBytes>>>(sum.get(),input4.get(),L); 
	
	/* sanity check */
	// host output of sum 
	cudaMemcpy( h_sum.get(), sum.get(), 1*sizeof(int), cudaMemcpyDeviceToHost);  
	std::cout << std::endl << " *h_sum : " << *h_sum << " (1<<30)+3 : " << ((1<<30)+3) << std::endl; 
	}

	/* ********** END of more tests of \sum_{i=1}^L 1 = L ********** */

	/* ************************************************************ */
	/* ********** more tests of \sum_{i=1}^L i = L(L+1)/2 ********** */
	/* ***** L = 1<<15 = 2^15 test ***** */ 
	{
	L = (1<< 15); 
	// notice how we're only launching 1/4 of L threads
	N_x = min( MAX_BLOCKS, ((L/4 + M_x - 1)/ M_x)); 

	std::unique_ptr<int[],decltype(del_ints_lambda)> input5(nullptr,del_ints_lambda); 
	cudaMallocManaged((void**) &input5,L*sizeof(int)); 
	inc_kernel<<< min((L+M_x-1)/M_x,MAX_BLOCKS), M_x>>>(input5.get(),1,L);
	cudaMemset(sum.get(), 0,sizeof(int)); // reset the sum 
	
	sum_kernel<<<N_x,M_x,sharedBytes>>>(sum.get(),input5.get(),L); 
	
	/* sanity check */
	// host output of sum 
	cudaMemcpy( h_sum.get(), sum.get(), 1*sizeof(int), cudaMemcpyDeviceToHost);  
	std::cout << std::endl << " *h_sum : " << *h_sum << " L(L+1)/2 : " << (L*(L+1)/2) << std::endl; 
	}

	/* ***** L = 1<<15 + 2 = 2^15 +2 test ***** */ 
	{
	L = (1<< 15) + 2; 
	// notice how we're only launching 1/4 of L threads
	N_x = min( MAX_BLOCKS, ((L/4 + M_x - 1)/ M_x)); 

	std::unique_ptr<int[],decltype(del_ints_lambda)> input6(nullptr,del_ints_lambda); 
	cudaMallocManaged((void**) &input6,L*sizeof(int)); 
	inc_kernel<<< min((L+M_x-1)/M_x,MAX_BLOCKS), M_x>>>(input6.get(),1,L);
	cudaMemset(sum.get(), 0,sizeof(int)); // reset the sum 
	
	sum_kernel<<<N_x,M_x,sharedBytes>>>(sum.get(),input6.get(),L); 
	
	/* sanity check */
	// host output of sum 
	cudaMemcpy( h_sum.get(), sum.get(), 1*sizeof(int), cudaMemcpyDeviceToHost);  
	std::cout << std::endl << " *h_sum : " << *h_sum << " L(L+1)/2  : " << (L*(L+1)/2) << std::endl; 
	}




}
