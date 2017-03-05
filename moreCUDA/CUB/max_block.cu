/** \file max_block.cu
 * \author Ernest Yeung
 * \email  ernestyalumni@gmail.com
 * Demonstrates CUB usage of block reduction, with max use
 * cf. https://www.microway.com/hpc-tech-tips/introducing-cuda-unbound-cub/
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
 * nvcc -std=c++11 -D_MWAITXINTRIN_H_INCLUDED -lcurand -I./CUB/cub/ cuRAND_eg.cu -o cuRAND_eg.exe
 * 
 * Also note that I was on a GeForce GTX 980 Ti and CUDA Toolkit 8.0 with latest drivers, and so 
 * for my case, Compute or SM requirements was (very much) met
 * 
 * Compiling notes: -lcurand needed for #include <curand.h>, otherwise you get these kinds of errors:
 * tmpxft_00000d2f_00000000-4_cuRAND_eg.cudafe1.cpp:(.text+0x103): undefined reference to `curandCreateGenerator'
 * tmpxft_00000d2f_00000000-4_cuRAND_eg.cudafe1.cpp:(.text+0x114): undefined reference to `curandSetPseudoRandomGeneratorSeed'
 * tmpxft_00000d2f_00000000-4_cuRAND_eg.cudafe1.cpp:(.text+0x12b): undefined reference to `curandGenerateUniform'
 * Also, be careful of the order of compiling libraries in the future; I forgot where, but I read somewhere that order matters for which library
 * 
 * -D_MWAITXINTRIN_H_INCLUDED
 * needed, otherwise, obtain these errors:
 * /usr/lib/gcc/x86_64-redhat-linux/5.3.1/include/mwaitxintrin.h(36): error: identifier "__builtin_ia32_monitorx" is undefined
 * 
 * /usr/lib/gcc/x86_64-redhat-linux/5.3.1/include/mwaitxintrin.h(42): error: identifier "__builtin_ia32_mwaitx" is undefined
 * 
 **********************************************************************/
#include <iostream>
#include <vector>
#include <algorithm> // std::for_each

#include <cub/cub.cuh> // Supposedly, this is the same as <cub/block/block_reduce.cuh> (EY ???) // cub::BlockReduceAlgorithm, cub::BLOCK_REDUCE_RAKING
/* #include <cub/cub.cuh>
 * cub::BlockReduceAlgorithm 
 * cub::BLOCK_REDUCE_RAKING
 * */
//#include <cub/block/block_reduce.cuh> 
//#include <cub/block/block_load.cuh>
//#include <cub/block/block_store.cuh>

#include <curand.h> // curandGenerator_t, curandCreateGenerator

/* cf. https://nvlabs.github.io/cub/classcub_1_1_block_reduce.html
 * cub::BlockReduce< T, BLOCK_DIM_X, ALGORITHM, BLOCK_DIM_Y, BLOCK_DIM_Z, PTX_ARCH > Class Template Reference
 * template<
    typename T, 
    int BLOCK_DIM_X, 
    BlockReduceAlgorithm ALGORITHM = BLOCK_REDUCE_WARP_REDUCTIONS, 
    int BLOCK_DIM_Y = 1, 
    int BLOCK_DIM_Z = 1, 
    int PTX_ARCH = CUB_PTX_ARCH>

	class cub::BlockReduce< T, BLOCK_DIM_X, ALGORITHM, BLOCK_DIM_Y, BLOCK_DIM_Z, PTX_ARCH >
 * */


//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------
cub::CachingDeviceAllocator dev_allocator(true);  // Caching allocator for device memory


//---------------------------------------------------------------------
// Kernels
//---------------------------------------------------------------------


// Thread-block reduction - a simple CUB example, from https://www.microway.com/hpc-tech-tips/introducing-cuda-unbound-cub/
template <int M_x, cub::BlockReduceAlgorithm ALGORITHM>  // M_x is BLOCK_SIZE or number of threads in a single block
__global__ void maxKernel(int* d_max, int* d_input)
{
	int k_x = threadIdx.x + blockDim.x * blockIdx.x ; 
	
	// Specialize BlockReduce type for our thread block
	using BlockReduceT = cub::BlockReduce<int,M_x,ALGORITHM> ;
	
	// Allocate temporary storage in shared memory
	
	// In a template declaration, typename can be used as an alternative to class to declare type template parameters 
	// cf. http://en.cppreference.com/w/cpp/keyword/typename
	__shared__ typename BlockReduceT::TempStorage temp_storage ;
/* cf. https://nvlabs.github.io/cub/structcub_1_1_block_reduce_1_1_temp_storage.html
 * cub::BlockReduce< T, BLOCK_DIM_X, ALGORITHM, BLOCK_DIM_Y, BLOCK_DIM_Z, PTX_ARCH >::TempStorage Struct Reference
 * The operations exposed by BlockReduce require a temporary memory allocation of this nested type 
 * for thread communication. 
 * This opaque storage can be allocated directly using the __shared__ keyword. */
 
	int val = d_input[k_x];
	int block_max = BlockReduceT(temp_storage).Reduce(val,cub::Max());
	
	// update global max value
	if (threadIdx.x == 0) {
		atomicMax(d_max,block_max);
	}
	
	return;
}

// cf. https://www.microway.com/hpc-tech-tips/introducing-cuda-unbound-cub/ Optimizing performance by limiting concurrency
template <int VALS_PER_THREAD, 
			int BLOCK_SIZE, // aka M_x aka BLOCK_THREADS
			cub::BlockReduceAlgorithm ALGORITHM>
__global__ void maxKernel_opt(int* d_max, int* d_input)
{
	int k_x = threadIdx.x + blockDim.x * blockIdx.x;
	
	// Special BlockReduce type for our thread block
	using BlockReduceT = cub::BlockReduce<int, BLOCK_SIZE, ALGORITHM>;
	
	// Allocate temporary storage in shared memory
	__shared__ typename BlockReduceT::TempStorage temp_storage;
	
	// Assign multiple values to each block thread
	int val[VALS_PER_THREAD];
	
	// Code to initialize the val array has been omitted originally, so I'm guessing it's like this from 
	// example_block_reduce.cu
	cub::LoadDirectStriped<BLOCK_SIZE>(threadIdx.x, d_input, val);
	
	int block_max = BlockReduceT(temp_storage).Reduce(val,cub::Max());
	
	// update global max value
	if (threadIdx.x == 0) {
		atomicMax(d_max,block_max);
	}
	return;
}


//template <int BLOCK_THREADS, int ITEMS_PER_THREAD, BlockReduceAlgorithm ALGORITHM>
//__global__ void 

int main() {
	
	/* boilerplate */
	// initiate correct GPU
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	if (deviceCount == 0) {
		exit(EXIT_FAILURE);
	}
	int dev = 0;
	cudaSetDevice(dev);
	
	cudaDeviceProp devProps;
	if (cudaGetDeviceProperties(&devProps, dev) == 0) {
		std::cout << " Using device " << dev << ":\n" ;
		std::cout << devProps.name << "; global mem: " << (int)devProps.totalGlobalMem <<
			"; compute v" << (int)devProps.major << "." << (int)devProps.minor << "; clock: " <<
			(int)devProps.clockRate << " kHz" << std::endl; }
	// END if GPU properties

//	CubDebugExit( cudaDeviceSynchronize() );
	
	/**
	 *******************************************************************
	 * Further BOILERPLATE
	 ******************************************************************/
	// how many array entries?  n = ?  for block reduce, it maxes out at the hardware's max number of threads in a block
//	int N_x = 32; // BLOCK_THREADS
//	int M_x = 32; // ITEMS_PER_THREAD

	constexpr int n = 1024;  // "TILE_SIZE"
	
	 // Initialization (on host CPU)
	std::vector<int> x(n); // test a LARGE histogram; input values or observations 

	int j = 0;
	// see all the different ways to use for_each to initiate and increment, with for_each, a std::vector
	std::for_each(x.begin(), x.end(), [&j](int &value) { value = j++; });
	std::for_each(x.begin(), x.end(), [](int &n){ n++; });

	std::vector<int>::const_iterator it = x.begin();
	std::for_each(x.begin(), x.end(), [&it](int &value) { value = *(++it) ; } );
	
	// use std::random_shuffle to permute order of the elements 
	std::random_shuffle( x.begin(), x.end() ) ; 
	
	// sanity check
	std::cout << " max element of x : " << 
		*(std::max_element( x.begin(), x.end() ) ) << std::endl;

	
	// Allocate problem device arrays
	int 	*dev_dat = nullptr;  // NULL, otherwise for nullptr, you're going to need the -std=c++11 flag for compilation
	int   *dev_max;

//	int *dev_i_dat = nullptr; 
	
	/* Allocate n ints on device */
    CubDebugExit(dev_allocator.DeviceAllocate((void**)&dev_dat, sizeof(int) * (n) ));
    CubDebugExit(dev_allocator.DeviceAllocate((void**)&dev_max, sizeof(int)  ));

//    CubDebugExit(dev_allocator.DeviceAllocate((void**)&dev_i_dat, sizeof(int) * n  ));


	CubDebugExit(cudaMemcpy(dev_dat, x.data(), sizeof(int) * (n), cudaMemcpyHostToDevice));

	// Boilerplate to make random device arrays

	/* Create pseudo-random number generator */
//	curandGenerator_t gen;
//	curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT);
		
	/* Set seed */
//	curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
	
	/* Generate n ints on device  */
//	curandGeneratePoisson( gen, dev_i_dat, n , 1.); // lambda


	/** ******************************************************************** 
		* END of device GPU boilerplate
		* */

	
	maxKernel<n,cub::BLOCK_REDUCE_RAKING><<<1, n>>>( 
														dev_max, 
														dev_dat);
	
	// generate output array on host
	int h_max = 0;
	
	/* Copy device memory to host */
	CubDebugExit(cudaMemcpy(&h_max, dev_max, sizeof(int) , cudaMemcpyDeviceToHost));
	std::cout << "dev_max (after BlockReduce with BLOCK_REDUCE_RAKING) : " << h_max << std::endl;
	
	
	// Check for kernel errors and STDIO from the kernel, if any
	CubDebugExit( cudaPeekAtLastError());
	CubDebugExit( cudaDeviceSynchronize());

	maxKernel_opt<32,32,cub::BLOCK_REDUCE_WARP_REDUCTIONS><<<1,n>>>( dev_max, dev_dat) ;

	/* Copy device memory to host */
	CubDebugExit(cudaMemcpy(&h_max, dev_max, sizeof(int) , cudaMemcpyDeviceToHost));
	std::cout << "dev_max (after BlockReduce with BLOCK_REDUCE_WARP_REDUCTIONS) : " << h_max << std::endl;

	// Check for kernel errors and STDIO from the kernel, if any
	CubDebugExit( cudaPeekAtLastError());
	CubDebugExit( cudaDeviceSynchronize());
	
	
	/* Clean up */
    if (dev_dat) CubDebugExit(dev_allocator.DeviceFree(dev_dat));
	if (dev_max) CubDebugExit(dev_allocator.DeviceFree(dev_max));
//	if (dev_i_dat) CubDebugExit(dev_allocator.DeviceFree(dev_i_dat));


//	CubDebugExit( cudaDeviceReset() );

	return 0;
}








	
	
	
	
	
