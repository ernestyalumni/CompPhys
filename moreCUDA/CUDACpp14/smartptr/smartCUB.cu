/**
 * @file   : smartCUB.cu
 * @brief  : Smart pointers for CUB content/source file in CUDA C++14, 
 * @author : Ernest Yeung <ernestyalumni@gmail.com>
 * @date   : 20171007  
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
 * nvcc -std=c++14 -I ~/cub-1.7.4 -dc smartCUB.cu -o smartCUB.o
 * 
 * */
#include "smartCUB.h"

std::shared_ptr<float> reduce_Sum( const int Lx, std::shared_ptr<float> & d_in) {

	// Allocate device output array
	auto deleterRR=[&](float* ptr){ cudaFree(ptr); };
	std::shared_ptr<float> d_out(new float(0.f), deleterRR);
	cudaMallocManaged((void **) &d_out, 1*sizeof(float));
	
	// Request and allocate temporary storage
	std::shared_ptr<void> d_temp_storage(nullptr, deleterRR);
	size_t temp_storage_bytes=0;
	cub::DeviceReduce::Sum( d_temp_storage.get(), temp_storage_bytes, d_in.get(), d_out.get(), Lx);
	
	cudaMallocManaged((void**) &d_temp_storage, temp_storage_bytes);
	
	// Run
	cub::DeviceReduce::Sum(d_temp_storage.get(), temp_storage_bytes,d_in.get(), d_out.get(), Lx);
	
	return d_out;
}  


