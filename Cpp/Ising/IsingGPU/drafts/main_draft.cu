/**
 * @file   : main_draft.cu
 * @brief  : main file draft for 2-dim. Ising in CUDA C++11/14, 
 * @details : 
 * 
 * @author : Ernest Yeung <ernestyalumni@gmail.com>
 * @date   : 20180108    
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
 * nvcc main.cu ./devgrid2d/devgrid2d.cu -o main
 * 
 * */
#include <iostream>
#include <memory>  
#include <array> // std::array in TransProb struct
#include <cmath> // std::exp in TransProb


/* ********** device GPU structs ********** */

struct Sysparam {
	float E; // total energy E
	float M; // total magnetization H 
	float T; // temperature T of the system (it's kT; treat Boltzmann constant k as a unit conversion)  
}; 

struct Avg {
	// (data) members
	// average values of physical parameters  
	float Eavg; 
	float Mavg; 
	float Esq_avg; // Esq = E*E 
	float Msq_avg; // Msq = M*M
	float absM_avg; // absM = |M| 
	float M4_avg; // M4_avg = M*M*M*M
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
	float get_by_DeltaE(const int DeltaE) {
		return transProb[DeltaE+8]; } ;
};

// custom deleters as structs   
struct del_Sysparam_struct { void operator()(Sysparam* ptr) { cudaFree(ptr); } };
struct del_Avg_struct { void operator()(Avg* ptr) { cudaFree(ptr); } };	
struct del_TransProb_struct { void operator()(TransProb* ptr) { cudaFree(ptr); } };  	

/* ********** END of device GPU structs ********** */


/* ********** host CPU structs ********** */
struct h_Sysparam { 
	float E; // total energy E 
	float M; // total magnetization M 
	float T; // temperature T of the system 

	// constructors
	// default constructor
	/** @fn Sysparam()
	 * @brief default constructor for struct Sysparam 
	 * @details set all E,M,T parameters to 0
	 * */
	h_Sysparam() : E {0.f}, M {0.f}, T {0.f} {}; 

	/** @fn Sysparam(double, double,double)
	 * @brief constructor for struct Sysparam 
	 * */
	h_Sysparam(float E, float M, float T) : E {E}, M {M}, T {T} {} ;

	/** @fn Sysparam(double)
	 * @brief constructor for struct Sysparam, when only given the system temperature (initially)
	 * */
	h_Sysparam(float T) : E {0.f}, M {0.f}, T {T} {}; 	

};

struct h_Avg {
	// (data) members
	// average values of physical parameters  
	float Eavg; 
	float Mavg; 
	float Esq_avg; // Esq = E*E 
	float Msq_avg; // Msq = M*M
	float absM_avg; // absM = |M| 
	float M4_avg; // M4_avg = M*M*M*M
	
	// constructors
	// default constructor
	h_Avg() : Eavg {0.f}, Mavg(0.f), Esq_avg(0.f), Msq_avg{0.f}, absM_avg{0.f},M4_avg{0.f} {};

	h_Avg(float Eavg, float Mavg, float Esq_avg, float Msq_avg, float absM_avg, float M4_avg) : 
		Eavg {Eavg}, Mavg{Mavg}, Esq_avg{Esq_avg}, Msq_avg{Msq_avg}, absM_avg{absM_avg}, M4_avg{M4_avg}  { } ;

};


/** @struct TransProb
 *  @brief transition probabilities to new spin configuration for 2-dim. Ising model 
 * */
struct h_TransProb {
	// (data) members
	// transition probabilities data 
	std::array<float,17> transProb; 
	float J; // spin constant 
	
	// constructors
	// default constructor 
	h_TransProb() : J {1.f} { };

	h_TransProb(float J, h_Sysparam& sysparams) : J {J} {
		float T = sysparams.T; // temperature
		for (int de = -8; de<= 8; de+=4) { 
			transProb[de+8] = std::exp(-((float) de)/T); }
	}; 

	// getting functions
	/** @fn get_by_DeltaE 
	 * @details given DeltaE (\Delta E), DeltaE = -8J, -4J,...8J, we want to get the 
	 * transition probability from std::unique_ptr transprob (but transprob indexed by 
	 * 0,1,...(17-1)
	 * */
	float get_by_DeltaE(int DeltaE) {
		return transProb[DeltaE+8]; 
	}; 
};

/** @fn calc_transProb
 * 
 * */
void calc_transProb(TransProb & transProb, const float T) {
	for (int de = -8; de <= 8; de+=4) {
		transProb.transProb[de+8] = std::exp( -((float) de)/T); 
	}
}
 
/* ********** END of host CPU structs ********** */



int main(int argc, char* argv[]) 
{
	std::unique_ptr<Sysparam,del_Sysparam_struct> d_sysparams(nullptr, del_Sysparam_struct()); 
	cudaMallocManaged((void **) &d_sysparams, 1*sizeof(Sysparam));  
	std::unique_ptr<Avg,del_Avg_struct> d_avgs(nullptr, del_Avg_struct()); 
	cudaMallocManaged((void **) &d_avgs, 1*sizeof(Avg));  
	std::unique_ptr<TransProb,del_TransProb_struct> d_transProb(nullptr, del_TransProb_struct()); 
	cudaMallocManaged((void **) &d_transProb, 1*sizeof(TransProb));  

	/* ****************************************************************************************** */
	/* ******************** Ways to initialize structs on device GPU; 2 ways ******************** */
	/* ****************************************************************************************** */
	/* *************** 1. cudaMemcpy from host to device *************** */

	// error: have to be of same type
//	h_Sysparam h_sysparams { 1.f, 3.f, 2.f };  
//	h_Avg h_avgs { 1.1f, 2.1f, 1.2f, 2.2f, 2.3f, 2.4f }; 
//	h_TransProb(1.f, h_sysparams);  

	// some host CPU values to input 
	Sysparam h_sysparams { 1.f, 3.f, 2.f };  
	Avg h_avgs { 1.1f, 2.1f, 1.2f, 2.2f, 2.3f, 2.4f }; 
	TransProb h_transProb; 
	calc_transProb(h_transProb, h_sysparams.T); 

	cudaMemcpy( d_sysparams.get(), &h_sysparams, 1*sizeof(Sysparam), cudaMemcpyHostToDevice);  // possible error have to be of same type
	cudaMemcpy( d_avgs.get(), &h_avgs, 1*sizeof(Avg), cudaMemcpyHostToDevice);  // possible error have to be of same type
	cudaMemcpy( d_transProb.get(), &h_transProb, 1*sizeof(TransProb), cudaMemcpyHostToDevice);  // possible error have to be of same type

	/* sanity check; copy back to host for sanity check */  
	Sysparam h_sysparams_out ;  
	Avg h_avgs_out ; 
	TransProb h_transProb_out; 

	cudaMemcpy(&h_sysparams_out, d_sysparams.get(), 1*sizeof(Sysparam), cudaMemcpyDeviceToHost);  // possible error have to be of same type
	cudaMemcpy(&h_avgs_out, d_avgs.get(), 1*sizeof(Avg), cudaMemcpyDeviceToHost);  // possible error have to be of same type
	cudaMemcpy(&h_transProb_out, d_transProb.get(), 1*sizeof(TransProb), cudaMemcpyDeviceToHost);  // possible error have to be of same type
	
	std::cout << " h_sysparams_out : " << h_sysparams_out.E << " " << h_sysparams_out.M << " " << h_sysparams_out.T << std::endl; 
	std::cout << std::endl << " h_avgs_out : " << h_avgs_out.Eavg << " " << h_avgs_out.Mavg 
		<< " " << h_avgs_out.Esq_avg << " " << h_avgs_out.Msq_avg << " " << h_avgs_out.absM_avg << " " << h_avgs_out.M4_avg  << std::endl; 
	for (int de =-8; de <= 8; de+=4) {
		std::cout << h_transProb_out.transProb[de+8] << " "; } std::cout << std::endl;
	/* END of sanity check */ 

	/* *************** 2. directly from host set values *************** */
	d_sysparams->E = 1.5f; 
	d_sysparams->M = 3.5f; 
	d_sysparams->T = 2.5f; 
	cudaMemcpy(&h_sysparams_out, d_sysparams.get(), 1*sizeof(Sysparam), cudaMemcpyDeviceToHost);  // possible error have to be of same type
	std::cout << std::endl << " h_sysparams_out : " << h_sysparams_out.E << " " << h_sysparams_out.M << " " << h_sysparams_out.T << std::endl; 
	
	
	
	
}
