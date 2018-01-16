/**
 * @file   : sysparam.h
 * @brief  : Physical parameters of the system header file, in C++11/14, 
 * @details : as a struct, have total energy E, total magnetization M, temperature of entire system T (in energy units)  
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
#ifndef __SYSPARAM_H__  
#define __SYSPARAM_H__  

#include <memory>  
#include <cmath> // std::exp 

struct Sysparam { 
	double E; // total energy E 
	double M; // total magnetization M 
	double T; // temperature T of the system 

	// constructors
	Sysparam(); 	// default constructor
	Sysparam(double, double, double) ; 	
	Sysparam(double) ; 	// only given the temperature T of the system

};

struct Avg {
	// (data) members
	// average values of physical parameters  
	double Eavg; 
	double Mavg; 
	double Esq_avg; // Esq = E*E 
	double Msq_avg; // Msq = M*M
	double absM_avg; // absM = |M| 
	
	// constructors
	Avg(); // default constructor
	Avg(double,double,double,double,double); 
};

/** @struct TransProb
 *  @brief transition probabilities to new spin configuration for 2-dim. Ising model 
 * */
struct TransProb {
	// (data) members
	// transition probabilities data 
	std::unique_ptr<double[]> transprob; 
	double J; // spin constant 
	
	// constructors
	TransProb(); // default constructor
	TransProb(double, Sysparam& ); 
	
	// getting functions
	/** @fn get_by_DeltaE 
	 * @details given DeltaE (\Delta E), DeltaE = -8J, -4J,...8J, we want to get the 
	 * transition probability from std::unique_ptr transprob (but transprob indexed by 
	 * 0,1,...(17-1)
	 * */
	double get_by_DeltaE(int); 
};






#endif // __SYSPARAM_H__
