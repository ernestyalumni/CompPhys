/**
 * @file   : output.h
 * @brief  : process output after Metropolis algorithm header file, in C++11/14, 
 * @details : process output after Metropolis algorithm, in particular, struct Avg 
 * 
 * @author : Ernest Yeung <ernestyalumni@gmail.com>
 * @date   : 20171231    
 * @ref    : M. Hjorth-Jensen, Computational Physics, University of Oslo (2015)  
 * https://github.com/CompPhysics/ComputationalPhysics/blob/master/doc/Programs/LecturePrograms/programs/StatPhys/cpp/ising_2dim.cpp  
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
#ifndef __OUTPUT_H__
#define __OUTPUT_H__

#include "../grid2d/sysparam.h" // Sysparam, Avg 

#include <string> // std::string  
#include <fstream> // std::fstream
#include <iostream> // std::cout 

/** @fn process_avgs  
 * @brief process the averages, in struct Avg, for output 
 * @details this is needed, because we'll need to normalize or divide by total number of cycles, trials  
 * */ 
void process_avgs(const int, size_t, std::string&, Sysparam&, Avg&) ; 

#endif // __OUTPUT_H__  
