/**
 * @file   : main_smartptr.cu
 * @brief  : Main file for Smart pointers (shared and unique ptrs) classes, in C++14, 
 * @details : A playground to try out things in smart pointers; 
 * 				especially abstracting our use of smart pointers with CUDA.  
 * 				Notice that std::make_unique DOES NOT have a custom deleter! (!!!)
 * 				Same with std::make_shared!  
 * 			cf. https://stackoverflow.com/questions/34243367/how-to-pass-deleter-to-make-shared
 * @author : Ernest Yeung <ernestyalumni@gmail.com>
 * @date   : 20170904  
 * @ref    : cf. Scott Meyers Effective Modern C++
 * 				http://shaharmike.com/cpp/unique-ptr/
 * 			https://katyscode.wordpress.com/2012/10/04/c11-using-stdunique_ptr-as-a-class-member-initialization-move-semantics-and-custom-deleters/
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
 * nvcc -std=c++14 ./smartptr/smartptr.cu main_smartptr.cu -o main_smartptr.exe
 * 
 * */
#include <iostream> // std::cout 
#include <algorithm> // std::fill

#include "smartptr/smartptr.h"

int main(int argc, char* argv[]) {
	constexpr const size_t Lx = {1<<5}; // 2^5 = 32

	// Allocate host arrays
	std::vector<float> h_vec(Lx,1.f);
	std::fill(h_vec.begin()+1,h_vec.begin() + h_vec.size()/4, 3.f);
	std::fill(h_vec.begin()+h_vec.size()/4,h_vec.end() - h_vec.size()/5, 8.f);


//	auto u_instance = make_uniq_u( Lx) ; // error: cannot deduce the return type of function "make_uniq_u"

	RRModule R( Lx);

	R.load_from_hvec(h_vec);
	
	// Readout result into output array on host
	std::vector<float> out_vec(Lx,0.f);
	R.load_from_d_X(out_vec);
	std::cout << " \n for unique_ptr on device : " << std::endl;
	for (auto ele : out_vec) {
		std::cout << " " << ele ; 
	}
	std::cout << std::endl;

	// shared version of RRModule
	RRModule_sh R_sh( Lx);

	R_sh.load_from_hvec(h_vec);
	R_sh.load_from_d_X(out_vec);
	std::cout << " \n for shared_ptr on device : " << std::endl;
	for (auto ele : out_vec) {
		std::cout << " " << ele ; 
	}
	std::cout << std::endl;


	/*
	 * CUB
	 * */
	// Allocate host arrays
	std::vector<float> f_vec(Lx,1.f);
	RRModule d_in(Lx);
	d_in.load_from_hvec(f_vec);
	
	// Allocate device output array
//	auto make_sh_u(1);


	// try moving pointers around  
	RRModule sourceX(Lx);
	sourceX.load_from_hvec(h_vec);
	RRModule_sh targetX( Lx);
	auto targetX_ptr = std::move( targetX.get() );
	

	// Clean up 
	cudaDeviceReset();
	return 0;


}
