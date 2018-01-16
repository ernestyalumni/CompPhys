/**
 * @file   : grid2d.h
 * @brief  : 2-dim. grid with spins "sitting" or "living" on top of it, header file, in C++11/14, 
 * @details : struct with smart ptrs, unique ptrs
 * 
 * @author : Ernest Yeung <ernestyalumni@gmail.com>
 * @date   : 20171229    
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
#include <array> 	// std::array  

struct Spins2d {
	// (data) members
	// spin data
	std::unique_ptr<int[]> S; // spins
	// dimensions describing the data 
	std::array<size_t,2> L_is; // L_i = (L_x,L_y); use size_t instead of int because size_t is "bigger" 
	size_t L; // L = L_x*L_y 
	
	// constructors 
	Spins2d() = default; 	// default constructor
	Spins2d(std::array<size_t,2> & L_is) ; 

	// getting functions
	/** @fn entry 
	 * @details ROW-major ordering based, i.e. it's the elements in a single row that are contiguous in memory, 
	 * NOT the column entries  
	 * */
	int entry(int i, int j); 
		
};

#endif // END of __GRID2D_H__
