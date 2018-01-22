/**
 * @file   : grid2d.h
 * @brief  : 2-dim. grid with spins "sitting" or "living" on top of it, header file, in CUDA C++11/14, 
 * @details : struct with smart ptrs, unique ptrs
 * 
 * @author : Ernest Yeung <ernestyalumni@gmail.com>
 * @date   : 20171231    
 * @ref    : M. Hjorth-Jensen, Computational Physics, University of Oslo (2015)
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
#ifndef __GRID2D_H__  
#define __GRID2D_H__  

#include <memory> 	// std::unique_ptr std::make_unique  
#include <array>  	// std::array  

/* =============== custom deleters =============== */

/**
 * @brief custom deleter as a struct 
 * @details The function call operator () can be overloaded for objects of class type. 
 * When you overload ( ), you are not creating a new way to call a function. 
 * Rather, you are creating an operator function that can be passed an arbitrary number of parameters.
 * @ref https://www.tutorialspoint.com/cplusplus/function_call_operator_overloading.htm
 * */ 
struct deleterZZ_struct
{
	void operator()(int* ptr) const 
	{
		cudaFree(ptr);
	}
};

/** @struct deleterRR_struct
 * @details RR stands for real numbers */
struct deleterRR_struct
{
	void operator()(float* ptr) const 
	{
		cudaFree(ptr);
	}
};


/* =============== END of custom deleters =============== */


struct Spins2d {
	// (data) members
	// spin data
	std::unique_ptr<int[], deleterZZ_struct> S; // spins
	// dimensions describing the data 
	std::array<size_t,2> L_is; // L_i = (L_x,L_y); use size_t instead of int because size_t is "bigger" 
	float J; 

	// default constructor
	Spins2d();  
	
	// constructors 
	Spins2d(std::array<size_t,2> & L_is) ; 

	Spins2d(std::array<size_t,2> & L_is, const float J) ; 


	// move constructor; necessitated by unique_ptr
	Spins2d(Spins2d &&) ; 
	
	// operator overload assignment = 
	Spins2d &operator=(Spins2d &&) ; 
};

/** @fn entry
 *  @brief takes (x,y) 2-dim. grid coordinates and "flattens" them (flatten functor) 
 *  to 1-dim. memory layout on global GPU memory  
 * @param i - i = 0,1,...Lx-1
 * @param j - j = 0,1,...Ly-1
 * @param Lx 
 * @return size_t k = i + j*Lx = 0,1... Lx*Ly-1
 * */
__device__ size_t entry(size_t i, size_t j, size_t Lx) ; 


/* energy E as local energy at each site */ 


/** @struct E2d
 * @brief energy in 2d 
 * */
struct DeltaE2d {
	// (data) members
	// (local) energy data
	std::unique_ptr<float[], deleterRR_struct> DeltaE; // local energy
	// dimensions describing the data 
	std::array<size_t,2> L_is; // L_i = (L_x,L_y); use size_t instead of int because size_t is "bigger" 

	// default constructor
	DeltaE2d();  
	
	// constructors 
	DeltaE2d(std::array<size_t,2> & L_is) ; 

	// move constructor; necessitated by unique_ptr
	DeltaE2d(DeltaE2d &&) ; 
	
	// operator overload assignment = 
	DeltaE2d &operator=(DeltaE2d &&) ; 
};



#endif // END of __GRID2D_H__
