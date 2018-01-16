/**
 * @file   : sysparam.cpp
 * @brief  : Physical parameters of the system separate/implementation file, in C++11/14, 
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
#include "./sysparam.h"  

// constructors
// default constructor
/** @fn Sysparam()
 * @brief default constructor for struct Sysparam 
 * @details set all E,M,T parameters to 0
 * */
Sysparam::Sysparam() : E {0.}, M {0.}, T {0.} {} 	

/** @fn Sysparam(double, double,double)
 * @brief constructor for struct Sysparam 
 * */
Sysparam::Sysparam(double E, double M, double T) : 
	E {E}, M {M}, T {T} {} 	

/** @fn Sysparam(double)
 * @brief constructor for struct Sysparam, when only given the system temperature (initially)
 * */
Sysparam::Sysparam(double T) : 
	E {0.}, M {0.}, T {T} {} 	


// default constructor
Avg::Avg() : Eavg {0.}, Mavg(0.), Esq_avg(0.), Msq_avg{0.}, absM_avg{0.} {}

Avg::Avg(double Eavg, double Mavg, double Esq_avg, double Msq_avg, double absM_avg) : 
	Eavg {Eavg}, Mavg{Mavg}, Esq_avg{Esq_avg}, Msq_avg{Msq_avg}, absM_avg{absM_avg} { }

// default constructor 
TransProb::TransProb() : J {1.} { 
	transprob = std::make_unique<double[]>(17); 
}

TransProb::TransProb(double J, Sysparam& sysparams) : J {J} {
	transprob = std::make_unique<double[]>(17); 
	double T = sysparams.T; // temperature
	for (int de = -8; de<= 8; de+=4) { 
		transprob[de+8] = std::exp(-((double) de)/T); }
	
}

// getting functions
/** @fn get_by_DeltaE 
 * @details given DeltaE (\Delta E), DeltaE = -8J, -4J,...8J, we want to get the 
 * transition probability from std::unique_ptr transprob (but transprob indexed by 
 * 0,1,...(17-1)
 * */
double TransProb::get_by_DeltaE(int DeltaE) {
	return transprob[DeltaE+8]; 
} 
