/**
 * @file   : cg_eg2.cu
 * @brief  : Examples of using cooperative groups
 * @details : cooperative groups for CUDA examples
 *  
 * @author : Ernest Yeung <ernestyalumni@gmail.com>
 * @date   : 20170107      
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
 * nvcc cg_eg2.cu -o cg_eg2
 * 
 * */
#include <cooperative_groups.h>  
#include <iostream>
#include <algorithm>  // std::fill_n
#include <memory>  // std::unique_ptr  

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


namespace cg = cooperative_groups;  

__device__ int reduce_sum(cg::thread_group g, int *temp, int val) 
{
	int lane = g.thread_rank(); 
	
	// Each iteration halves the number of active threads
	// Each thread adds its partial sum[i] to sum[lane+i] 
	for (int i = g.size() / 2; i >0; i/=2) 
	{
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
 * @brief compute many partial sums in parallel, 
 * @details compute many partial sums in parallel, 
 * 	where each thread strides through the array computing a partial sum 
 * */ 
__device__ int thread_sum(int *input, int n) 
{
	int sum =0;
	
	for (int i=threadIdx.x + blockDim.x * blockIdx.x; 
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
//	cg::thread_block g = cg::this_thread
	int block_sum = reduce_sum(g,temp,my_sum); 
	
	if (g.thread_rank() == 0) {
		atomicAdd(sum, block_sum); 
	}
};  

/** @fn thread_sum_gen
 * @brief compute many partial sums in parallel, Generalized to when n is not a power of 2
 * @details compute many partial sums in parallel, Generalized to when n is not a power of, 
 * 	where each thread strides through the array computing a partial sum 
 * */ 
__device__ int thread_sum_gen(int *input, int n) 
{
	int sum =0;

	unsigned int k_x = threadIdx.x + blockDim.x*blockIdx.x; 
	
	/* increment by blockDim.x*gridDim.x, so that a single thread will do all the 
	 * "work" needed done on n, especially if n >= gridDim.x*blockDim.x = N_x*M_x */	
	for (int i=k_x; 
			i < n/4; 
			i += blockDim.x * gridDim.x) 
	{
		int4 in = ((int4*) input)[i]; 
		sum += in.x + in.y + in.z + in.w; 
	}
	// process remaining elements
//	for (unsigned int idx= k_x + n/4*4; idx < n; idx += threadIdx.x + blockIdx.x *blockDim.x) {
	for (unsigned int idx= k_x + n/4*4; idx < n; idx += 4 ) {
		sum += input[idx];
	}
	
	return sum; 
};  


/** @fn whatisthread_sum
 * @brief Explicitly step through what thread_sum above does  */
__global__ void whatisthread_sum(int *input, int *output, int n) 
{
	int my_sum = thread_sum(input,n);
	unsigned int k_x = threadIdx.x + blockDim.x * blockIdx.x; 
	output[k_x] = my_sum;  
//	printf("%d ", my_sum); 
}

/** @fn whatisreduce_sum
 * @brief Explicitly step through what reduce_sum above does */
//__global__ void whatisreduce_sum(int *data, int* , int val) 



/** @fn sum_kernel_block_gen
 * @brief sum kernel by blocks, generalized for n not a power of 2 */
__global__ void sum_kernel_block_gen(int *sum, int *input, int n)
{
	int my_sum = thread_sum_gen(input, n); 
	
	extern __shared__ int temp[];
	auto g = cg::this_thread_block();
	int block_sum = reduce_sum(g,temp,my_sum); 
	
	if (g.thread_rank() == 0) {
		atomicAdd(sum, block_sum); 
	}
};  

__global__ void reduce_sum_test(int* output, int* temp, int val) {
	unsigned int k_x = threadIdx.x + blockDim.x * blockIdx.x; 
	cg::thread_block g = cg::this_thread_block(); 
	int block_sum = reduce_sum(g,temp,val);
	
	// store back the output for this 1 thread
	output[k_x] = block_sum; 
};


int main(int argc, char* argv[]) 
{
	size_t MAXGRIDSIZE = get_maxGridSize();  
	/* ***** (thread) grid,block dims ***** */ 
	/* min of N_x, number of (thread) blocks on grid in x-direction, and MAX_BLOCKS allowed is 
	 * determined here */
	constexpr const unsigned int M_x = 1<<6;  // M_x = number of threads in x-direction, in a single block, i.e. blocksize; 2^8=256  
	int n = 1<<7; // doesn't output correct values for n = 1<<30    
	const unsigned int MAX_BLOCKS = (MAXGRIDSIZE + M_x - 1)/ M_x; 
	const unsigned int N_x = min( MAX_BLOCKS, ((n + M_x - 1)/ M_x)); 
	int sharedBytes = M_x * sizeof(int);  
	/* ***** END of (thread) grid,block dims ***** */ 
	

	std::cout << " N_x : " << N_x << std::endl; 
	
	// setup input, output 
	auto del_ints_lambda_main=[&](int* ptr) { cudaFree(ptr); }; 
	std::unique_ptr<int,decltype(del_ints_lambda_main)> sum(nullptr,del_ints_lambda_main); 
	std::unique_ptr<int[],decltype(del_ints_lambda_main)> input(nullptr,del_ints_lambda_main); 
	std::unique_ptr<int[],decltype(del_ints_lambda_main)> output(nullptr,del_ints_lambda_main); 
	cudaMallocManaged((void**)&sum, sizeof(int)) ;
	cudaMallocManaged((void**)&input, n*sizeof(int));
	cudaMallocManaged((void**)&output, n*sizeof(int)); 
	std::fill_n(input.get(),n,1); 
	cudaMemset(sum.get(), 0,sizeof(int));
	cudaMemset(output.get(),0, n * sizeof(int)); 

	cudaDeviceSynchronize();
	whatisthread_sum<<<N_x,M_x>>>(input.get(), output.get(), n);
	cudaDeviceSynchronize();
	
	
	/* sanity check*/
	// host output array of ints
	std::unique_ptr<int[]> h_output = std::make_unique<int[]>( n );  
	cudaMemcpy( h_output.get(), output.get(), n*sizeof(int), cudaMemcpyDeviceToHost);  
	for (int idx=0; idx< min(n/2,256); idx++) { std::cout << h_output[idx] << " " ; }
	std::cout << std::endl << std::endl; 
	for (int idx=n-min(n/2,256); idx<n; idx++) { std::cout << h_output[idx] << " " ; }
//	for (int idx=0; idx<100; idx++) { std::cout << output[idx] << " " ; }  // Segmentation Fault
//	for (int idx=0; idx<100; idx++) { std::cout << input[idx] << " " ; }	// Segmentation Fault  

	/* ********** (thread) grid, block dimensions ********** */
	unsigned int L = 1<<5; // 1<<8=256, 1<<5=32
	unsigned int Mx = 1<<5; // blockSize, 1<<6=64
	unsigned int Nx = (L+Mx-1)/Mx; // number of (thread) blocks on the grid  
	sharedBytes = Mx * sizeof(int);  
	
	std::unique_ptr<int[],decltype(del_ints_lambda_main)> outputs32(nullptr,del_ints_lambda_main); 
	cudaMallocManaged((void**) &outputs32,L*sizeof(int)); 
	
	std::unique_ptr<int[],decltype(del_ints_lambda_main)> temp32(nullptr,del_ints_lambda_main); 
	cudaMallocManaged((void**) &temp32, Mx*sizeof(int));  
	
	cudaMemset(temp32.get(),0,sizeof(int)); 
	cudaMemset(outputs32.get(),0,sizeof(int));
	
	reduce_sum_test<<<Nx,Mx>>>(outputs32.get(),temp32.get(),1); 

	/* sanity check*/
	// host output array of ints
	std::unique_ptr<int[]> h_outputs32 = std::make_unique<int[]>( L );  
	cudaMemcpy( h_outputs32.get(), outputs32.get(), L*sizeof(int), cudaMemcpyDeviceToHost);  
	std::cout << std::endl << std::endl; 
	for (int idx=0; idx< min(L/2,256); idx++) { std::cout << h_outputs32[idx] << " " ; }
	std::cout << std::endl << std::endl; 
//	for (int idx=n-min(L/2,256); idx<L; idx++) { std::cout << h_outputs32[idx] << " " ; }
	for (int idx=L-min(L/2,256); idx<L; idx++) { std::cout << h_outputs32[idx] << " " ; }
	std::cout << std::endl << std::endl; 


	/* ***** check summation \sum_{i=0}^L i = L(L+1)/2 ***** */	
	/* *******************************************************/
	// L = 32 case
	std::unique_ptr<int[],decltype(del_ints_lambda_main)> input1(nullptr,del_ints_lambda_main); 
	cudaMallocManaged((void**) &input1,L*sizeof(int)); 
	inc_kernel<<<Nx,Mx>>>(input1.get(),1,L);
	
	sum_kernel_block_gen<<<Nx,Mx,sharedBytes>>>(sum.get(),input1.get(),L); 
	/* sanity check*/
	// host output array of ints
	std::unique_ptr<int> h_sum = std::make_unique<int>( 0 );  
	cudaMemcpy( h_sum.get(), sum.get(), 1*sizeof(int), cudaMemcpyDeviceToHost);  
	std::cout << std::endl << " *h_sum : " << *h_sum << " (L*(L+1))/2 : " << (L*(L+1))/2 << std::endl; 
	
	// L = 64 case  
	L = 1 << 6 ; 
	Nx = (L+Mx-1)/Mx; // number of (thread) blocks on the grid  
	cudaMemset(sum.get(), 0,sizeof(int));
	
	std::unique_ptr<int[],decltype(del_ints_lambda_main)> input2(nullptr,del_ints_lambda_main); 
	cudaMallocManaged((void**) &input2,L*sizeof(int)); 
	inc_kernel<<<Nx,Mx>>>(input2.get(),1,L);
	
	sum_kernel_block_gen<<<Nx,Mx,sharedBytes>>>(sum.get(),input2.get(),L); 

	/* sanity check*/
	// host output array of ints
	cudaMemcpy( h_sum.get(), sum.get(), 1*sizeof(int), cudaMemcpyDeviceToHost);  
	std::cout << std::endl << " *h_sum : " << *h_sum << " (L*(L+1))/2 : " << (L*(L+1))/2 << std::endl; 
	
	// L = 128 case  
	L = 1 << 7 ; 
	Nx = (L+Mx-1)/Mx; // number of (thread) blocks on the grid  
	cudaMemset(sum.get(), 0,sizeof(int));
	
	std::unique_ptr<int[],decltype(del_ints_lambda_main)> input3(nullptr,del_ints_lambda_main); 
	cudaMallocManaged((void**) &input3,L*sizeof(int)); 
	inc_kernel<<<Nx,Mx>>>(input3.get(),1,L);
	
	sum_kernel_block_gen<<<Nx,Mx,sharedBytes>>>(sum.get(),input3.get(),L); 

	/* sanity check*/
	// host output array of ints
	cudaMemcpy( h_sum.get(), sum.get(), 1*sizeof(int), cudaMemcpyDeviceToHost);  
	std::cout << std::endl << " *h_sum : " << *h_sum << " (L*(L+1))/2 : " << (L*(L+1))/2 << std::endl; 
	
	for (int i=24; i<38 ; i++ ) {
		std::cout << " i : " << i << " i/4 : " << i/4 << " i/4 * 4 : " << i/4*4 << " " << std::endl; }
	
}
