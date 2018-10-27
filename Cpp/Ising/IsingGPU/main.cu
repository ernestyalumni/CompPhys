/**
 * @file   : main.cpp
 * @brief  : main driver file for 2-dim. Ising in CUDA C++11/14, 
 * @details : 
 * 
 * @author : Ernest Yeung <ernestyalumni@gmail.com>
 * @date   : 20180103    
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
 * nvcc main.cu ./grid2d/grid2d.cu ./grid2d/sysparam.cu ./dynam/XORMRGgens.cu ./dynam/metropolis.cu ./common/gridsetup.cu -o main
 * 
 * */
#include "./grid2d/grid2d.h"  // Spins2d (struct)  
#include "./grid2d/sysparam.h"  // Sysparam, Avg, TransProb, Sysparam_ptr, Avg_ptr, TransProb_ptr, constTransProb
#include "./dynam/metropolis.h" // initialize_allup, metropolis
#include "./common/gridsetup.h" // get_maxGridSize()
#include "./FileIO/output.h" 	// process_avgs 

#include <iostream>
#include <chrono>  

int main(int argc, char* argv[]) 
{

	// number of spins, related to 2-dim. grid size Lx x Ly
	std::array<size_t, 2> L_is { 1<<10, 1<<10 }; // 1<<10 = 1024
	
	Spins2d spins = {L_is};  

	std::cout << " L : " << spins.L_is[0]*spins.L_is[1] << std::endl; 
	
	// number of trials or number of times to run the Metropolis algorithm 
	constexpr const unsigned int trials = 50000;
	
	// file name
	std::string filename = "./data/IsingMetroGPU.bin";  

	
	/* ***** (thread) grid,block dims ***** */ 
	/* min of N_x, number of (thread) blocks on grid in x-direction, and MAX_BLOCKS allowed is 
	 * determined here */
	size_t MAXGRIDSIZE = get_maxGridSize();  
	auto MAXGRIDSIZES = get_maxGridSizes();
	std::cout << " MAXGRIDSIZES : " << MAXGRIDSIZES[0] << std::endl; 
	
	// (thread) block dims., remember max. no. threads per block is 1024, as of compute capability 5.2
	dim3 M_is { 1<<5, 1<<5 }; 

	// (thread) grid, block dims for curandstates and other 1-dim. arrays  
	unsigned int M_x = 1<<8; // 2^8 = 256 
	unsigned long MAX_BLOCKS = (MAXGRIDSIZE + M_x - 1)/ M_x; 
	unsigned int N_x = min( MAX_BLOCKS, (( spins.L_is[0]*spins.L_is[1] + M_x - 1)/ M_x)); 
	/* ***** END of (thread) grid,block dims ***** */ 


	// physical parameters on the host CPU, used for File I/O
	Sysparam h_sysparams_out;
	Avg h_avgs_out;

	constexpr const float initial_temp = 1.f;  // typically 1.
	constexpr const float final_temp = 3.f;  // typically 3.
	constexpr const float tempstep = 0.005f;  // typically 0.05

/*	Sysparam_ptr sysparams_ptr = { initial_temp } ;
	TransProb_ptr transprob_ptr = { initial_temp , 1.f } ;
	Avg_ptr avgs_ptr;  */
	
	/* ERROR warning: if you move this line for devStatesXOR "earlier", bus error obtained.  
	 * 
	 * */
	// since curand_init calls are slow, do it once for the grid from the host main code
//	devStatesXOR devstatesXOR = { spins.L_is[0]*spins.L_is[1], N_x,M_x }; 

	// following 1 line is for timing code purposes only 
	auto start = std::chrono::steady_clock::now();

	for (float temp=initial_temp; temp<=final_temp; temp+=tempstep) {
		// following 1 line is for timing code purposes only 
		auto start1 = std::chrono::steady_clock::now();

		Sysparam_ptr sysparams_ptr = { temp } ;
		TransProb_ptr transprob_ptr = { temp , 1.f } ;
		Avg_ptr avgs_ptr;  

		/* ERROR warning: if you move this line for devStatesXOR "earlier", bus error obtained.  
		 * 
		 * */
		// since curand_init calls are slow, do it once for the grid from the host main code
		devStatesXOR devstatesXOR = { spins.L_is[0]*spins.L_is[1], N_x,M_x }; 


		initialize_allup(spins,sysparams_ptr, MAXGRIDSIZES, M_is);

		metropolis(spins,sysparams_ptr,avgs_ptr,transprob_ptr,MAXGRIDSIZES,M_is,devstatesXOR,trials); 

		// following 3 line is for timing code purposes only 
		auto end1 = std::chrono::steady_clock::now();
		auto diff1 = end1-start1;
		std::cout << std::chrono::duration<double,std::milli>(diff1).count() << " ms" << std::endl;  

		cudaMemcpy(&h_sysparams_out,sysparams_ptr.d_sysparams.get(),1*sizeof(Sysparam),cudaMemcpyDeviceToHost);  
		cudaMemcpy(&h_avgs_out,avgs_ptr.d_avgs.get(),1*sizeof(Avg),cudaMemcpyDeviceToHost);  

		process_avgs(trials, spins.L_is[0]*L_is[1], filename, h_sysparams_out, h_avgs_out) ;  
			
		// following 3 line is for timing code purposes only 
		auto end2 = std::chrono::steady_clock::now();
		auto diff2 = end2-start1;
		std::cout << std::chrono::duration<double,std::milli>(diff2).count() << " ms" << std::endl;  
			
	}
			

	// following 3 line is for timing code purposes only 
	auto end = std::chrono::steady_clock::now();
	auto diff = end-start;
	std::cout << std::chrono::duration<double,std::milli>(diff).count() << " ms" << std::endl;  


}


	/* sanity check */
/*	Sysparam h_sysparams_out ;  
	cudaMemcpy(&h_sysparams_out, sysparams_ptr.d_sysparams.get(), 1*sizeof(Sysparam), cudaMemcpyDeviceToHost);  // possible error have to be of same type
	std::cout << " h_sysparams_out : " << h_sysparams_out.E << " " << h_sysparams_out.M << " " 
		<< h_sysparams_out.T << std::endl; 
*/
