/**
 * @file   : device_reduce_unified.cu
 * @brief  : demonstrate CUB and CUDA Unified Memory (management)  
 * uses CUDA Unified Memory (Management)
 * @author : Ernest Yeung	ernestyalumni@gmail.com
 * @date   : 20170417
 * @ref    :  cf. https://nvlabs.github.io/cub/example_device_reduce_8cu-example.html
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
  * CUDA Programming note: 
  * cf. http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-unified-memory-programming-hd
  * "It is not permitted for the CPU to access any managed allocations or variables while 
  * the GPU is active for devices with concurrentManagedAccess property set to 0.  
  * On these systems, concurrent CPU/GPU accesses, even to different managed memory 
  * allocations, will cause a segmentation fault because the page is considered 
  * inaccessible to the CPU.  
  * 
  * This code runs successfully on devices of compute capability 6.x due to the GPU page faulting capability which lifts 
  * all restrictions on simultaneous access.  However, such memory access is invalid on pre-6.x architectures 
  * even though the CPU is accessing different data than the GPU.  
  * 
  * EY : 20170418 I'm on a 1050 mobile.  
  * */
// COMPILATION TIP:
// nvcc -std=c++11 -arch='sm_61' device_reduce_unified.cu -o device_reduce_unified.exe
// nvcc -std=c++11 -arch='sm_61' -I./cub-1.6.4/ device_reduce_unified.cu -o device_reduce_unified.exe

#include <iostream>

#include <cub/cub.cuh>

constexpr const	int Lx = 1500;

// Allocate arrays
__device__ __managed__ float f[Lx];
__device__ __managed__ float g;  

//---------------------------------------------------------------------
// Test generation
//---------------------------------------------------------------------

// n is the exponential power, and so we're doing f(k_x)^n
template <typename TypeT> 
__global__ void Initialize(TypeT* f, const int n, const int Lx) {
	int k_x = threadIdx.x + blockDim.x * blockIdx.x ;
	if (k_x >= Lx) { return; }
	
	f[k_x] = powf( ((TypeT) k_x), n) ;
}

//

//---------------------------------------------------------------------
// Main
//---------------------------------------------------------------------

int main(int argc, char** argv) 
{
	dim3 M_i(1024,1,1);
	
	Initialize<float><<< (Lx+M_i.x-1)/M_i.x, M_i.x >>>( f,1,Lx);
	
	// Request and allocate temporary storage
	size_t temp_storage_bytes =0;
	float* d_temp_storage=nullptr;

	cub::DeviceReduce::Sum( d_temp_storage, temp_storage_bytes, f, &g, Lx );

	std::cout << " temp_storage_bytes : " << temp_storage_bytes << std::endl;
	CubDebugExit( 
		cudaMalloc(&d_temp_storage, temp_storage_bytes) );

	// Run
	cub::DeviceReduce::Sum( d_temp_storage, temp_storage_bytes, f, &g, Lx) ;

	// print out
	std::cout << " g : " << g << std::endl;

	// Cleanup
	if (d_temp_storage) { 
		CubDebugExit( cudaFree( d_temp_storage) ); }
		
}
