/**
 * @file   : devgrid2d.cpp
 * @brief  : 2-dim. grid with spins "sitting" or "living" on top of it, separate implementation file, in CUDA C++11/14, 
 * @details : struct with smart ptrs, unique ptrs  
 * 
 * @author : Ernest Yeung <ernestyalumni@gmail.com>
 * @date   : 20171229    
 * @ref    : Ch. 8 Structures, Unions, and Enumerations; Bjarne Stroustrup, The C++ Programming Language, 4th Ed.  
 * Addison-Wesley 
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
 * g++ main.cpp ./structs/structs.cpp -o main
 * 
 * */
#include "./devgrid2d.h"  

// constructor  
Spins2d::Spins2d(std::array<size_t,2> & L_is) : L_is {L_is} { 
	L = L_is[0] * L_is[1];  

	std::unique_ptr<int[], deleterZZ_struct> d_S(new int[L]); 
	cudaMallocManaged((void **) &d_S, L * sizeof(int)) ;
	S = std::move(d_S);  
	
} 
