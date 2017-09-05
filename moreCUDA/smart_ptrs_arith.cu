/**
 * @file   : smart_ptrs_arith.cu
 * @brief  : Smart pointers (shared and unique ptrs) arithmetic, in C++11, 
 * 			 for cudaMalloc, cudaMemcpy
 * @author : Ernest Yeung <ernestyalumni@gmail.com>
 * @date   : 20170904  
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
#include <iostream> // std::cout 
#include <memory>  // std::shared_ptr, std::unique_ptr 

#include <vector> 	// std::vector

int main(int argc, char* argv[]) {
	constexpr const size_t Lx = {1<<5};  // 2^5 = 32

	// Allocate host arrays
	std::vector<float> f_vec(Lx/2,1.f);
    std::shared_ptr<float> sp(new float[Lx/2],std::default_delete<float[]>());

    // "boilerplate" initialization of interesting values on host
    for (auto iptr = sp.get(); iptr != sp.get() + Lx/2; iptr++) {
        *iptr = 11.f * ((int) (iptr - sp.get()));
    }
	// sanity check
	std::cout << "\n sanity check for shared_ptr sp on host : " << std::endl; 
	for (int idx =0; idx < Lx/2; idx++) {
		std::cout << idx << " : " << sp.get()[idx] << ", "; 
	}

	/*
	 *  shared_ptr on the device GPU
	 */

	// Allocate problem device arrays
	auto deleter=[&](float* ptr){ cudaFree(ptr); };
	std::shared_ptr<float> d_sh_in(new float[Lx], deleter);
	cudaMalloc((void **) &d_sh_in, Lx * sizeof(float));

	cudaMemcpy(d_sh_in.get(), f_vec.data(), Lx/2*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(d_sh_in.get()+Lx/2, sp.get(), Lx/2*sizeof(float),cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();

	// Readout result into output array on host
	std::vector<float> out_vec(Lx,0.f);
	cudaMemcpy(out_vec.data(), d_sh_in.get(), Lx*sizeof(float),cudaMemcpyDeviceToHost);
	
	std::cout << " \n for shared_ptr on device : " << std::endl;
	for (auto ele : out_vec) {
		std::cout << " " << ele ; 
	}
	std::cout << std::endl;
	

	/*
	 *  unique_ptr on the device GPU
	 */

	// device pointers
	std::unique_ptr<float[], decltype(deleter)> d_u_in(new float[Lx], deleter);
	cudaMalloc((void **) &d_u_in, Lx * sizeof(float));

	cudaMemcpy(d_u_in.get(), sp.get(), Lx/2*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(d_u_in.get()+Lx/2, f_vec.data(), Lx/2*sizeof(float),cudaMemcpyHostToDevice);

	// Readout result into output array on host
	cudaMemcpy(out_vec.data(), d_u_in.get(), Lx*sizeof(float),cudaMemcpyDeviceToHost);
	
	std::cout << " \n for unique_ptr on device : " << std::endl;
	for (auto ele : out_vec) {
		std::cout << " " << ele ; 
	}
	std::cout << std::endl;



	// Clean up
	cudaDeviceReset();
	return 0;
}
