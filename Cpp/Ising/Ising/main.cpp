/**
 * @file   : main.cpp
 * @brief  : main driver file for 2-dim. Ising in C++11/14, 
 * @details : 
 * 
 * @author : Ernest Yeung <ernestyalumni@gmail.com>
 * @date   : 20171229    
 * @ref    : M. Hjorth-Jensen, Computational Physics, University of Oslo (2015)
 * https://github.com/CompPhysics/ComputationalPhysics/blob/master/doc/Programs/LecturePrograms/programs/StatPhys/cpp/ising_2dim.cpp
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
 * g++ main.cpp -o main
 * 
 * */
#include "./boundary/boundary.h"  // periodic
#include "./grid2d/grid2d.h"  // Spins2d (struct)  
#include "./grid2d/sysparam.h"  // Sysparam, Avg, TransProb
#include "./dynam/metropolis.h" // initialize
#include "./FileIO/output.h" 	// process_avgs 

#include <chrono>  

int main(int argc, char* argv[]) {
	
	long idum = -1; // random long seed for ran1 in metropolis.h
	
	std::string filename = "./data/IsingMetroCPU.bin";  
	
	constexpr const double initial_temp = 1.;  // typically 1.
	constexpr const double final_temp = 3.;  // typically 3.
	constexpr const double tempstep = 0.05;  // typically 0.05

	
	constexpr const int trials = 100000; // trials is number of trials to run Metropolis algorithm // typically 1000000 
	
	// number of spins, related to 2-dim. grid size Lx x Ly 
	std::array<size_t, 2> L_is { 128, 128 } ;  
	
	Spins2d spins = {L_is}; 


	// following 1 line is for timing code purposes only 
	auto start = std::chrono::steady_clock::now();

	for (double temp=initial_temp; temp<=final_temp; temp+=tempstep) {
		
		// initialize energy, magnetization and temperature
		Sysparam sysparams = { temp }; 

		// setup struct, array for possible energy changes  
		TransProb transprob = { 1., sysparams }; 
	
		// initialize struct for expectation values 
		Avg average;
		initialize( spins, sysparams); 
	
		// start MonteCarlo computation
		for (int cycles = 1; cycles <= trials; cycles++) { 
			Metropolis(idum, spins, sysparams, transprob, average); 
		}
		// print results
		process_avgs(trials, spins.L, filename, sysparams, average); 
	
	}
	
	// following 3 line is for timing code purposes only 
	auto end = std::chrono::steady_clock::now();
	auto diff = end-start;
	std::cout << std::chrono::duration<double,std::milli>(diff).count() << " ms" << std::endl;  
	
}
