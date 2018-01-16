/**
 * @file   : sysparam.h
 * @brief  : Physical parameters of the system header file, in CUDA C++11/14, on device GPU 
 * @details : as a struct, have total energy E, total magnetization M, temperature of entire system T (in energy units)  
 * 
 * @author : Ernest Yeung <ernestyalumni@gmail.com>
 * @date   : 20180103    
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
 * nvcc main.cpp ./structs/structs.cpp -o main
 * 
 * */
#ifndef __SYSPARAM_H__  
#define __SYSPARAM_H__  

#include <array>  // std::array
#include <math.h> // std::exp 
#include <memory> // std::unique_ptr

/* =============== device GPU structs =============== */ 

struct Sysparam { 
	float E; // total energy E 
	float M; // total magnetization M 
	float T; // temperature T of the system 
};

struct Avg {
	// (data) members
	// average values of physical parameters  
	float Eavg; 
	float Mavg; 
	float Esq_avg; // Esq = E*E 
	float Msq_avg; // Msq = M*M
	float absM_avg; // absM = |M| 
	float M4_avg; // M4 = M*M*M*M
};

/** @struct TransProb
 *  @brief transition probabilities to new spin configuration for 2-dim. Ising model 
 * */
struct TransProb {
	// (data) members
	// transition probabilities data 
	std::array<float,17> transProb;
	float J; // spin constant 
	
	
	// getting functions
	/** @fn get_by_DeltaE 
	 * @details given DeltaE (\Delta E), DeltaE = -8J, -4J,...8J, we want to get the 
	 * transition probability from std::unique_ptr transprob (but transprob indexed by 
	 * 0,1,...(17-1)
	 * */
	float get_by_DeltaE(const int); 
};

/* =============== custom deleters =============== */

// custom deleters as structs   
struct del_Sysparam_struct { void operator()(Sysparam* ptr) { cudaFree(ptr); } };
struct del_Avg_struct { void operator()(Avg* ptr) { cudaFree(ptr); } };	
struct del_TransProb_struct { void operator()(TransProb* ptr) { cudaFree(ptr); } };  	

/* =============== END of custom deleters =============== */

/* struct of structs
 * I chose this design because 
 * 1. I want a struct such that its data members is identified by pseudo-mathematical symbols, 
 * i.e. a data member E should denote total Energy  
 * 1.a. I want this struct to be "empty" in that it has no custom constructors/destructors, 
 * so that it can be used by both host CPU and device GPU, and can be qualified with __constant__, so to be 
 * put in constant memory 
 * 2. I want a struct of these structs so to automate its construction/destruction: 
 * cudaMallocManaged, cudaFree, and use smart pointers for it
 * */

struct Sysparam_ptr { 
	// (data) members
	std::unique_ptr<Sysparam,del_Sysparam_struct> d_sysparams; // Sysparam, E,M,T 
	
	// default constructor
	Sysparam_ptr(); 
	
	// constructors
	Sysparam_ptr(const float, const float, const float);
	Sysparam_ptr(const float); // only given the temperature T of the system  
	
	// move constructor; necessitated by unique_ptr
	Sysparam_ptr(Sysparam_ptr &&) ;  
 
	// operator overload assignment = 
	Sysparam_ptr &operator=(Sysparam_ptr &&) ;
}; 
 
struct Avg_ptr { 
	// (data) members
	std::unique_ptr<Avg,del_Avg_struct> d_avgs; // Sysparam, E,M,T 
	
	// default constructor
	Avg_ptr(); 
	
	// constructors
	Avg_ptr(const float,const float,const float,const float,const float,const float);
	
	// move constructor; necessitated by unique_ptr
	Avg_ptr(Avg_ptr &&) ;  
 
	// operator overload assignment = 
	Avg_ptr &operator=(Avg_ptr &&) ;
}; 

struct TransProb_ptr { 
	// (data) members
	std::unique_ptr<TransProb,del_TransProb_struct> d_transProb; // Sysparam, E,M,T 
	
	// default constructor
	TransProb_ptr(); 
	
	// constructors
	TransProb_ptr(const float T,const float J);
	
	// move constructor; necessitated by unique_ptr
	TransProb_ptr(TransProb_ptr &&) ;  
 
	// operator overload assignment = 
	TransProb_ptr &operator=(TransProb_ptr &&) ;
}; 

/* ********** END of device GPU structs ********** */
//__constant__ TransProb constTransProb; // TransProb struct instant in header (hdr) file, constant __constant__ memory



#endif // END of __SYSPARAM_H__  
