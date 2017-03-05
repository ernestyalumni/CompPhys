/** \file block_reduce.cu
 * \author Ernest Yeung
 * \email  ernestyalumni@gmail.com
 * \brief Demonstrates reduce, on a block, i.e. single (thread) block, for CUB
 * 
 * If you find this code useful, feel free to donate directly and easily at this direct PayPal link: 
 * 
 * https://www.paypal.com/cgi-bin/webscr?cmd=_donations&business=ernestsaveschristmas%2bpaypal%40gmail%2ecom&lc=US&item_name=ernestyalumni&currency_code=USD&bn=PP%2dDonationsBF%3abtn_donateCC_LG%2egif%3aNonHosted 
 * 
 * which won't go through a 3rd. party such as indiegogo, kickstarter, patreon.  
 * Otherwise, I receive emails and messages on how all my (free) material on physics, math, and engineering have 
 * helped students with their studies, and I know what it's like to not have money as a student, but love physics 
 * (or math, sciences, etc.), so I am committed to keeping all my material open-source and free, whether or not 
 * sufficiently crowdfunded, under the open-source MIT license: 
 * 	feel free to copy, edit, paste, make your own versions, share, use as you wish.    
 * Peace out, never give up! -EY
 * 
 * */
/**
 * Compilation tips
 * 
 *  ** EY : 20170303
 * But here's what I did, on a GeForce GTX 980 Ti
 * I wanted to include the cub library, which is in a separate folder (I downloaded, unzipped)
 * but it's not symbolically linked from root (I REALLY don't want to mess with root directory right now).
 * so I put the folder (straight downloaded from the internet) into a desired, arbitrary location;
 * in this case, I put it in `./` so that it's `./cub/`
 * Then I used the include flag -I
 * in the following manner:
 * nvcc -std=c++11 -lcurand -D_MWAITXINTRIN_H_INCLUDED -I./cub/ block_reduce.cu -o block_reduce.exe
 * 
 * Also note that I was on a GeForce GTX 980 Ti and CUDA Toolkit 8.0 with latest drivers, and so 
 * for my case, Compute or SM requirements was (very much) met
 * 
 * -D_MWAITXINTRIN_H_INCLUDED
 * needed for #include <algorithm>, otherwise, obtain these errors:
 * /usr/lib/gcc/x86_64-redhat-linux/5.3.1/include/mwaitxintrin.h(36): error: identifier "__builtin_ia32_monitorx" is undefined
 * 
 * /usr/lib/gcc/x86_64-redhat-linux/5.3.1/include/mwaitxintrin.h(42): error: identifier "__builtin_ia32_mwaitx" is undefined
 * 
 **********************************************************************/
#include <cub/cub.cuh> // cub::BlockReduceAlgorithm, cub::BLOCK_REDUCE_RAKING, cub::BLOCK_REDUCE

/* includes for boilerplate, to make interesting test values */
#include <iostream>
#include <vector>
#include <algorithm> // std::max_element
#include <numeric> // std:accumulate

#include <curand.h> // curandGenerator_t, curandCreateGenerator, 

//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------
cub::CachingDeviceAllocator dev_allocator(true);  // Caching allocator for device memory


//---------------------------------------------------------------------
// Kernels
//---------------------------------------------------------------------
/**
 * block-wide max over float
 * M_x = total number of threads in a single (thread) block, i.e. BLOCK_THREADS
 * d   = dimension of output, i.e. \mathbb{R}^d, i.e. ITEMS_PER_THREAD
 * */
template <int M_x, int d, cub::BlockReduceAlgorithm ALGORITHM>
__global__ void BlockMax1DKernel(float *dev_F, float *dev_maxF) {
	
	// Specialize BlockReduce type for our thread block
	using BlockReduceT = cub::BlockReduce<float,M_x,ALGORITHM>; 
	
	// Shared memory
	__shared__ typename BlockReduceT::TempStorage temp_storage;
	
	// Per-thread tile data
	float data[d];
	cub::LoadDirectStriped<M_x>(threadIdx.x, dev_F, data);
	
	// Compute max
	float aggregator = BlockReduceT(temp_storage).Reduce(data,cub::Max());

	// store max
	if (threadIdx.x == 0)
	{
		*dev_maxF = aggregator;
	}
}

template <int M_x, int d, cub::BlockReduceAlgorithm ALGORITHM>
__global__ void BlockSum1DKernel(float *dev_F, float *dev_out) {
	
	// Specialize BlockReduce type for our thread block
	using BlockReduceT = cub::BlockReduce<float,M_x,ALGORITHM>; 
	
	// Shared memory
	__shared__ typename BlockReduceT::TempStorage temp_storage;
	
	// Per-thread tile data
	float data[d];
	cub::LoadDirectStriped<M_x>(threadIdx.x, dev_F, data);
	
	// Compute max
	float aggregator = BlockReduceT(temp_storage).Sum(data);

	// store max
	if (threadIdx.x == 0)
	{
		*dev_out = aggregator;
	}
}


int main(int argc, char *argv[]) {
	
	/* boilerplate */
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	if (deviceCount == 0) {
		exit(EXIT_FAILURE);
	}
	int dev = 0;
	cudaSetDevice(dev);

	////////////////////////////////////////////////////////////////////
	// boilerplate for interesting values to test out for reduce ///////
	////////////////////////////////////////////////////////////////////
	constexpr int L1 = 1024; // "TILE_SIZE"
	constexpr int L2 = 2 * L1; // "TILE_SIZE"
	
	
	// Allocate problem device arrays
    float 	*dev_f = nullptr;  // NULL, otherwise for nullptr, you're going to need the -std=c++11 flag for compilation
	float   *dev_fmax = nullptr;
	float   *dev_fsum = nullptr;

	float   *dev_f2 = nullptr;
	
	/* Allocate n floats on device */
    CubDebugExit(dev_allocator.DeviceAllocate((void**)&dev_f, sizeof(float) * L1 ));

    CubDebugExit(dev_allocator.DeviceAllocate((void**)&dev_f2, sizeof(float) * L2 ));


	/* Allocate floats on device */
    CubDebugExit(dev_allocator.DeviceAllocate((void**)&dev_fmax, sizeof(float) ));
    CubDebugExit(dev_allocator.DeviceAllocate((void**)&dev_fsum, sizeof(float) ));

	/* Create pseudo-random number generator */
	curandGenerator_t gen;
	curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT);

	/* Set seed */
	curandSetPseudoRandomGeneratorSeed(gen,1234ULL);
	
	/* Generate n floats on device */
	constexpr float mean1 = 0.f;  // set this MANUALLY
	constexpr float stddev1 = 1.f; // set this MANUALLY
	curandGenerateNormal(gen, dev_f, L1, mean1,stddev1); // mean, then stddev

	curandGenerateLogNormal(gen, dev_f2, L2, mean1,stddev1); // mean, then stddev

	// generate output array on host
	std::vector<float> f_vec(L1,0.f);
	float h_max = 0;
	float h_sum = 0;

	std::vector<float> f2_vec(L2,0.f);

	////////////////////////////////////////////////////////////////////
	// END of boilerplate for interesting values to test out for reduce 
	////////////////////////////////////////////////////////////////////

	////////////////////////////////////////////////////////////////////
	//////////////////// BLOCK, GRID DIMENSIONS ////////////////////////
	constexpr int gridsize = 1; // EY : 20170304 for number of blocks on a grid, it seems like it's 1 for block reduce, which wasn't explicitly said so, but seemed implied in the CUB documentation
	constexpr int M_x1 = L1;
	constexpr int d1   = 1;

	constexpr int M_x2 = L1;
	constexpr int d2   = 2;
	
	///////////// END of BLOCK, GRID DIMENSIONS ////////////////////////
	////////////////////////////////////////////////////////////////////

	BlockMax1DKernel<M_x1,d1,cub::BLOCK_REDUCE_WARP_REDUCTIONS><<<gridsize,M_x1>>>(dev_f, dev_fmax);

	// Check for kernel errors and STDIO from the kernel, if any
	CubDebugExit( cudaPeekAtLastError());
	CubDebugExit( cudaDeviceSynchronize());

	BlockSum1DKernel<M_x1,d1,cub::BLOCK_REDUCE_WARP_REDUCTIONS><<<gridsize,M_x1>>>(dev_f, dev_fsum);

	/* Copy device memory to host */
	CubDebugExit(cudaMemcpy(f_vec.data(), dev_f, sizeof(float) * L1 , cudaMemcpyDeviceToHost));
	CubDebugExit(cudaMemcpy(&h_max, dev_fmax, sizeof(float) , cudaMemcpyDeviceToHost));
	std::cout << "dev_fmax (after BlockReduce with BLOCK_REDUCE_WARP_REDUCTIONS) : " << h_max << std::endl;

	CubDebugExit(cudaMemcpy(&h_sum, dev_fsum, sizeof(float) , cudaMemcpyDeviceToHost));
	std::cout << "dev_fsum (after BlockReduce with BLOCK_REDUCE_WARP_REDUCTIONS) : " << h_sum << std::endl;
	
	// sanity check
	std::cout << " max element of f_vec : " << 
		*(std::max_element( f_vec.begin(), f_vec.end() ) ) << std::endl;
	
	float f_sum_result = std::accumulate( f_vec.begin(), f_vec.end(), 0.f);
	std::cout << " summation on f_vec : " << f_sum_result << std::endl;
	
	// Check for kernel errors and STDIO from the kernel, if any
	CubDebugExit( cudaPeekAtLastError());
	CubDebugExit( cudaDeviceSynchronize());

	BlockMax1DKernel<M_x2,d2,cub::BLOCK_REDUCE_RAKING><<<gridsize,M_x2>>>(dev_f2, dev_fmax);

	CubDebugExit(cudaMemcpy(&h_max, dev_fmax, sizeof(float) , cudaMemcpyDeviceToHost));
	std::cout << "dev_fmax (after BlockReduce with BLOCK_REDUCE_RAKING) : " << h_max << std::endl;

	// Check for kernel errors and STDIO from the kernel, if any
	CubDebugExit( cudaPeekAtLastError());
	CubDebugExit( cudaDeviceSynchronize());

	BlockSum1DKernel<M_x1,d2,cub::BLOCK_REDUCE_RAKING><<<gridsize,M_x2>>>(dev_f2, dev_fsum);

	CubDebugExit(cudaMemcpy(&h_sum, dev_fsum, sizeof(float) , cudaMemcpyDeviceToHost));
	std::cout << "dev_fsum (after BlockReduce with BLOCK_REDUCE_RAKING) : " << h_sum << std::endl;


	// sanity check
	CubDebugExit(cudaMemcpy(f2_vec.data(), dev_f2, sizeof(float) * L2 , cudaMemcpyDeviceToHost));
	std::cout << " max element of f2_vec : " << 
		*(std::max_element( f2_vec.begin(), f2_vec.end() ) ) << std::endl;
	
	float f2_sum_result = std::accumulate( f2_vec.begin(), f2_vec.end(), 0.f);
	std::cout << " summation on f2_vec : " << f2_sum_result << std::endl;


	
	/* Clean up */
    if (dev_f) CubDebugExit(dev_allocator.DeviceFree(dev_f));
    if (dev_fmax) CubDebugExit(dev_allocator.DeviceFree(dev_fmax));
    if (dev_fsum) CubDebugExit(dev_allocator.DeviceFree(dev_fsum));

    if (dev_f2) CubDebugExit(dev_allocator.DeviceFree(dev_f2));

		
}



