/**
 * @file   : main_smartCUB.cu
 * @brief  : Main file for Smart pointers for CUB (shared and unique ptrs) classes, in C++14, 
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
 * nvcc -std=c++14 -I ~/cub-1.7.4 ./smartptr/smartCUB.cu ./smartptr/smartptr.cu main_smartCUB.cu -o main_smartCUB.exe
 * 
 * */

#include <iostream> // std::cout 

#include "smartptr/smartptr.h"
#include "smartptr/smartCUB.h"

int main(int argc, char* argv[]) {
	constexpr const int Lx = (1 << 8);
	std::cout << " Lx : " << Lx << std::endl;

	// Allocate host arrays
	std::vector<float> f_vec(Lx,1.f);
	
	// Allocate problem device arrays
	RRModule_sh X_sh(Lx);

    // Initialize device input
	X_sh.load_from_hvec(f_vec);

	auto d_in = std::move( X_sh.get() );
	auto d_out = reduce_Sum(Lx, d_in);

	// Allocate output host array
	std::vector<float> g_vec(1,0.f);

	// Copy results from Device to Host
	cudaMemcpy(g_vec.data(), d_out.get(), 1*sizeof(float),cudaMemcpyDeviceToHost);

	// print out result:
	std::cout << " g_vec[0] : " << g_vec[0] << std::endl;

	// Clean up
	cudaDeviceReset();
	return 0;


}
