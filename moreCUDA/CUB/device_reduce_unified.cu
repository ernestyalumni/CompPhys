/**
 * @file   : device_reduce.cu
 * @brief  : demonstrating usage of CUDA CUB's device-(wide) reduce or sum
 * uses CUDA Unified Memory (Management)
 * @author : Ernest Yeung	ernestyalumni@gmail.com
 * @date   : 20170418
 * @ref : cf. http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared
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
 /**
  * COMPILATION TIP(s)
  * 
  * nvcc -std=c++11 example_device_reduce.cu-o example_device_reduce.exe -I./cub/
  * but for __managed__ variables, you'll need to specify gpu-architecture or gpu compute capability:
  * nvcc -std=c++11 -arch='sm_52' device_reduce.cu -I./cub/ -o device_reduce.exe
  * */

#include <iostream>

#include <cub/cub.cuh>

constexpr const int Lx = 2000;

__device__ __managed__ float f[Lx]; 	// Lx size float array
__device__ __managed__ float g; // (single) float

//---------------------------------------------------------------------
// Test generation
//---------------------------------------------------------------------
template <typename TypeT>
__global__ void Initialize(TypeT* f, const int n, const int Lx) {
	int k_x = threadIdx.x + blockDim.x * blockIdx.x;
	if (k_x >= Lx) { return; }
	
	f[k_x] = powf( ((TypeT) k_x), n); 
}

//---------------------------------------------------------------------
// Main
//---------------------------------------------------------------------
int main(int argc, char** argv) 
{
	constexpr const int exp_power = 2;
	dim3 M_i(1024,1,1);

	Initialize<float><<< (Lx+M_i.x-1)/M_i.x, M_i.x >>>( f,exp_power,Lx);
	
	// Request and allocate temporary storage
	size_t temp_storage_bytes=0;
	float* d_temp_storage=nullptr;
	
	cub::DeviceReduce::Sum( d_temp_storage, temp_storage_bytes, f, &g, Lx);
	CubDebugExit( 
		cudaMalloc(&d_temp_storage, temp_storage_bytes) );
	
	// Run
	cub::DeviceReduce::Sum( d_temp_storage, temp_storage_bytes, f, &g , Lx); 
	
	// print out
	// You can only do the following line on a GPU card with compute capability of 6.x or greater
	// on the EVGA GeForce GTX 980Ti I have with Maxwell architecture, I can't do the following line of code, 
	// printing g immediately.  But I tested this exact same code on a 1050 mobile (Pascal architecture) and it works without a hitch
	// std::cout << " summation : " << g << std::endl; 
	// On compute capability 5.2 or so, less than 6.x, I have to do this:
	float h_g =0;
	cudaMemcpy(&h_g,&g,sizeof(float) * 1, cudaMemcpyDeviceToHost) ;

	std::cout << " summation : " << h_g << std::endl; 
	
	
	
	// Cleanup
	if (d_temp_storage) {
		CubDebugExit( cudaFree( d_temp_storage) ); }

	
}
	
