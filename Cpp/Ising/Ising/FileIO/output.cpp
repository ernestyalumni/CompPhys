/**
 * @file   : output.cpp
 * @brief  : process output after Metropolis algorithm separate/implementation file, in C++11/14, 
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
#include "./output.h"  

/** @fn process_avgs  
 * @brief process the averages, in struct Avg, for output 
 * @details this is needed, because we'll need to normalize or divide by total number of cycles, trials  
 * @param N - total number of spins 
 * */ 
void process_avgs(const int trials, size_t N, std::string& filename, Sysparam& sysparams, Avg& avgs) 
{
	double norm = 1./((double) trials) ; 
	double temp = sysparams.T; 

	// all expectation values are per spin, divided by 1/N
	
	double Eavg = avgs.Eavg * norm ; 
	double Esq_avg = avgs.Esq_avg * norm; 
	double Mavg = avgs.Mavg * norm ; 
	double Msq_avg = avgs.Msq_avg * norm; 
	double absM_avg = avgs.absM_avg * norm  ; 
	
	double Evariance = (Esq_avg - Eavg*Eavg)/((double) N); 
	double Mvariance = (Msq_avg - Mavg*Mavg)/((double) N);  

	Eavg = Eavg / ((double) N); 
	Mavg = Mavg / ((double) N); 
	absM_avg = absM_avg  / ((double) N); 

	std::fstream fstre(filename, fstre.binary | fstre.app | fstre.in | fstre.out );
	
	if (!fstre.is_open()) {
		std::cout << "Failed to open " << filename << std::endl; 
	} else {
		fstre.write(reinterpret_cast<char*>(&temp), sizeof temp); // binary output  
		fstre.write(reinterpret_cast<char*>(&Eavg), sizeof Eavg); 
		fstre.write(reinterpret_cast<char*>(&Evariance), sizeof Evariance); 
		fstre.write(reinterpret_cast<char*>(&Mavg), sizeof Mavg); 
		fstre.write(reinterpret_cast<char*>(&Mvariance), sizeof Mvariance);
		fstre.write(reinterpret_cast<char*>(&absM_avg), sizeof absM_avg); 
		fstre.close();
	}	
}
