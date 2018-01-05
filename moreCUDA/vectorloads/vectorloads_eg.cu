/**
 * @file   : vectorloads_eg.cu
 * @brief  : Examples of vector loads for higher memory access efficiency 
 * @details : use vector loads and stores to increase bandwidth utilization   
 *  
 * @author : Ernest Yeung <ernestyalumni@gmail.com>
 * @date   : 20170104      
 * @ref    : https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-increase-performance-with-vectorized-memory-access/
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
 * nvcc vectorloads_eg.cu -o vectorloads_eg
 * DEBUG/ASSEMBLY TIP
 * cuobjdump -sass vectorloads_eg
 * */
#include <memory> // std::unique_ptr
#include <algorithm> // std::fill_n

/* ********** old, naive way of simple memory copy kernel ********** */
__global__ void device_copy_scalar_kernel(int* d_in, int* d_out, int N) 
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x; 
	for (int i=idx ; i < N; i += blockDim.x * gridDim.x ) {
		d_out[i] = d_in[i]; 
	}
}

/** @fn device_copy_scalar 
 * @brief driver function for CUDA kernel device_copy_scalar_kernel 
 * */
void device_copy_scalar(int* d_in, int* d_out, int N, unsigned int MAX_BLOCKS	) 
{
	int threads = 128; 
	int blocks = min((N+threads-1)/threads, MAX_BLOCKS);  
	device_copy_scalar_kernel<<<blocks,threads>>>(d_in,d_out,N);  
}

/* ********** use vector loads ********** */
__global__ void device_copy_vector2_kernel(int* d_in, int* d_out, int N) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x ; 

	// executes only N/2 times because each iteration processes only 2 elements
	for (int i=idx; i < N/2; i += blockDim.x * gridDim.x ) 
	{
		// use casting technique to use vectorized load and store 
		reinterpret_cast<int2*>(d_out)[i] = reinterpret_cast<int2*>(d_in)[i] ; 
	}
	
	// handle any remaining elements which may arise if N is not divisible by 2 
	// process remaining elements  
	for (int i=idx + N/2 *2; i<N; i += threadIdx.x + blockIdx.x*blockDim.x) {
		d_out[i] = d_in[i]; 
	}
}

void device_copy_vector2(int* d_in, int* d_out, int N, unsigned int MAX_BLOCKS) {
	int threads = 128; 
	int blocks=min((N/2+threads - 1)/threads, MAX_BLOCKS);
	
	// finally, launch half as many threads as we did in scalar kernel
	device_copy_vector2_kernel<<<blocks,threads>>>(d_in,d_out,N);  
}

__global__ void device_copy_vector4_kernel(int* d_in, int* d_out, int N) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x; 
	for (int i=idx; i < N/4 ; i+= blockDim.x*gridDim.x) {
		reinterpret_cast<int4*>(d_out)[i] = reinterpret_cast<int4*>(d_in)[i] ; 
	}
	
	// process remaining elements 
	for (int i = idx + N/4 * 4; i<N; i += threadIdx.x + blockIdx.x*blockDim.x) {
		d_out[i] = d_in[i];
	}
}

void device_copy_vector4(int* d_in, int* d_out, int N, unsigned int MAX_BLOCKS) {
	int threads = 128; 
	int blocks=min((N/4+threads-1)/ threads, MAX_BLOCKS); 
	
	device_copy_vector4_kernel<<<blocks,threads>>>(d_in,d_out,N); 
}

int main(int argc, char* argv[]) {

	cudaDeviceProp prop;
	int count;
	cudaGetDeviceCount(&count); 
	int MAXGRIDSIZE;
	if (count >0) {
		cudaGetDeviceProperties( &prop, 0); 
		MAXGRIDSIZE = prop.maxGridSize[0];
	} else { return EXIT_FAILURE; }

	/* ***** (thread) grid,block dims ***** */ 
	constexpr const int M_x = 128;  // M_x = number of threads in x-direction, in a single block
	const unsigned int MAX_BLOCKS = (MAXGRIDSIZE + M_x - 1)/ M_x; 

	constexpr const unsigned long long L { 1<<19 };  
	
	auto del_ZZarr_lambda =[&](int* ptr) { cudaFree(ptr); };
	std::unique_ptr<int[], decltype(del_ZZarr_lambda)> dev_in(nullptr, del_ZZarr_lambda); 
	cudaMallocManaged((void **)&dev_in, L * sizeof(int));  
	std::unique_ptr<int[], decltype(del_ZZarr_lambda)> dev_out(nullptr, del_ZZarr_lambda); 
	cudaMallocManaged((void **)&dev_out, L * sizeof(int));  

	/* ***** "boilerplate" test values ***** */
	std::fill_n(dev_in.get(), L, 2); 
	
	device_copy_scalar( dev_in.get(), dev_out.get(), L, MAX_BLOCKS);  

	device_copy_vector2(dev_in.get(), dev_out.get(), L, MAX_BLOCKS);  

	device_copy_vector4(dev_in.get(), dev_out.get(), L, MAX_BLOCKS);  

	
}
