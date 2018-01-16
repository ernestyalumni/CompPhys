/**
 * @file   : metropolis.h
 * @brief  : Metropolis algorithm for 2-dim. grid, with initialization, header file, in C++11/14, 
 * @details : initialize function, ran1, metropolis function
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
#ifndef __METROPOLIS_H__  
#define __METROPOLIS_H__  

#include "../grid2d/grid2d.h" // Spins2d struct
#include "../grid2d/sysparam.h" // Sysparam, Avg, TransProb structs 
#include "../boundary/boundary.h" // periodic

#include <cmath> // std::fabs

/** 
 * @fn ran1
 * @ref https://github.com/CompPhysics/ComputationalPhysics/blob/cde9f3b1ee798c36c66794bdd332b030a2c82c5c/doc/Programs/LecturePrograms/programs/cppLibrary/lib.cpp
 * @brief double ran1(long *idum)
 * @details is a "Minimal" random number generator of Park and Miller 
 * (see Numerical reciple page 280) with Bays-Durham shuffle and 
 * added safeguards.  Call with idum a negative integer to initialize; 
 * thereafter, do not alter idum between successive deviates in a 
 * sequence.  RNMX should approximate the largest floating point value 
 * that is less than 1. 
 * The function returns a uniform deviate between 0.0 and 1.0
 * (exclusive of end-point values).  
 */ 
double ran1(long *idum);


void initialize(Spins2d &, Sysparam& );

// The Metropolis algorithm 
void Metropolis(long &, Spins2d &, Sysparam &, TransProb&, Avg& );  


#endif // END of __METROPOLIS_H__
