/**
 * @file   : device_reduce_smart.cu
 * @brief  : demonstrating usage of CUDA CUB's device-(wide) reduce or sum, also with 
 * 			 C++11/14 smart pointers
 * @author : Ernest Yeung <ernestyalumni@gmail.com>
 * @date   : 20170902  
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
 * nvcc -std=c++11 -I ~/cub/cub-1.7.3 device_reduce_smart.cu -o device_reduce_smart.exe
 * 
 * */

#include <iostream>	// std::cout
#include <vector> 	// std::vector
#include <memory> 	// std::unique_ptr


#include <cub/cub.cuh>


//---------------------------------------------------------------------
// Main
//---------------------------------------------------------------------
int main(int argc, char* argv[]) {
	constexpr const int Lx = (1 << 8);
	std::cout << " Lx : " << Lx << std::endl;

	// Allocate host arrays
	std::vector<float> f_vec(Lx,1.f);
	
	// Allocate problem device arrays
	auto deleter=[&](float* ptr){ cudaFree(ptr); };
//	std::unique_ptr<float[], decltype(deleter)> d_in(new float[Lx], deleter);
	std::shared_ptr<float> d_in(new float[Lx], deleter);
	cudaMalloc((void **) &d_in, Lx * sizeof(float));


    // Initialize device input
	cudaMemcpy(d_in.get(), f_vec.data(), Lx*sizeof(float),cudaMemcpyHostToDevice);

	// Allocate device output array
//	std::unique_ptr<float, decltype(deleter)> d_out(new float(0.f), deleter);
	std::shared_ptr<float> d_out(new float(0.f), deleter);
	cudaMalloc((void **) &d_out, 1 * sizeof(float));


    // Request and allocate temporary storage
//	std::unique_ptr<void, decltype(deleter)> d_temp_storage(nullptr, deleter);
	std::shared_ptr<void> d_temp_storage(nullptr, deleter);
//	void* d_temp_storage = nullptr;
	
	size_t 		temp_storage_bytes = 0;

	cub::DeviceReduce::Sum( d_temp_storage.get(), temp_storage_bytes, d_in.get(),d_out.get(),Lx);

//	cudaMalloc( (void **) d_temp_storage.get(), temp_storage_bytes);
	cudaMalloc((void **) &d_temp_storage, temp_storage_bytes);
	
	// Run
	cub::DeviceReduce::Sum(d_temp_storage.get(),temp_storage_bytes,d_in.get(),d_out.get(),Lx);

	// Allocate output host array
	std::vector<float> g_vec(1,0.f);
	
	// Copy results from Device to Host
	cudaMemcpy(g_vec.data(), d_out.get(), 1*sizeof(float),cudaMemcpyDeviceToHost);

	// print out result:
	std::cout << " g_vec[0] : " << g_vec[0] << std::endl;

	// Clean up
//	cudaFree(d_temp_storage);
	cudaDeviceReset();
	return 0;
}
