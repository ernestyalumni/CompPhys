/**
 * @file   : grid2d.cpp
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
#include "./grid2d.h"  

// default constructor
Spins2d::Spins2d() : J {1.f} {}

// constructor  
Spins2d::Spins2d(std::array<size_t,2> & L_is) : L_is {L_is} , J {1.f} { 
//	L = L_is[0] * L_is[1];  

	std::unique_ptr<int[], deleterZZ_struct> d_S(nullptr, deleterZZ_struct() ); 
	cudaMallocManaged((void **) &d_S, L_is[0]*L_is[1] * sizeof(int)) ;
	S = std::move(d_S);  
} 

Spins2d::Spins2d(std::array<size_t,2> & L_is, const float J) : L_is {L_is} , J {J} { 
	std::unique_ptr<int[], deleterZZ_struct> d_S(nullptr, deleterZZ_struct() ); 
	cudaMallocManaged((void **) &d_S, L_is[0]*L_is[1] * sizeof(int)) ;
	S = std::move(d_S);  
} 


// move constructor 
Spins2d::Spins2d( Spins2d && old_spins2d) : 
//	L_is {old_spins2d.L_is}, L { old_spins2d.L }, 
	L_is {old_spins2d.L_is}, J { old_spins2d.J }, 
	S { std::move( old_spins2d.S ) } { }  
	
// operator overload assignment = 
Spins2d & Spins2d::operator=(Spins2d && old_spins2d) {
	L_is = old_spins2d.L_is;
//	L = old_spins2d.L; 
	J = old_spins2d.J; 
	S = std::move( old_spins2d.S ); 
	return *this;  
}


/** @fn entry
 *  @brief takes (x,y) 2-dim. grid coordinates and "flattens" them (flatten functor) 
 *  to 1-dim. memory layout on global GPU memory  
 * @param i - i = 0,1,...Lx-1
 * @param j - j = 0,1,...Ly-1
 * @param Lx 
 * @return size_t k = i + j*Lx = 0,1... Lx*Ly-1
 * */
__device__ size_t entry(size_t i, size_t j, size_t Lx) {
	return (i + j*Lx); 
}
