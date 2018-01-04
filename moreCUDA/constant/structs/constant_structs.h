/**
 * @file   : constant_structs.h
 * @brief  : Examples of using constant memory for CUDA, with smart pointers, in header file  
 * @details : constant memory for CUDA examples
 *  
 * @author : Ernest Yeung <ernestyalumni@gmail.com>
 * @date   : 20170103      
 * @ref    : http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-specifiers
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
 * nvcc main.cu -o main
 * 
 * */
 #ifndef __CONSTANT_STRUCTS_H__ 
 #define __CONSTANT_STRUCTS_H__ 
 
 #include <array> // std::array  
 
 struct S_arrs {
	 std::array<float,17> transProb; 
	 std::array<int, 5> sMask; 
	 
	 size_t Lx; 	// 8 byte
	 size_t Ly;		// 8 byte
	 unsigned long long Nx; 	// 8 byte
	 unsigned long long Ny; 	// 8 byte
};

__constant__ S_arrs constS_arrs_hdr; // S_arrs struct instance in header (hdr) file, constant __constant__ memory

 
#endif // END of __CONSTANT_STRUCTS_H__
