/**
 * @file   : XORMRGgens.h
 * @brief  : Examples using cuRAND device API to generate pseudorandom numbers using either XORWOW or MRG32k3a generators, header file   
 * @details : This program uses the device CURAND API.  The purpose of these examples is explore scope and compiling and modularity/separation issues with CURAND   
 * 
 * @author : Ernest Yeung <ernestyalumni@gmail.com>
 * @date   : 20180101      
 * @ref    : http://docs.nvidia.com/cuda/curand/device-api-overview.html#device-api-example
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
 * nvcc -lcurand -dc XORMRGgens2distri.cu -o XORMRGgens2distri  
 * */
 
#ifndef __XORMRGGENS_H__
#define __XORMRGGENS_H__  

#include <curand_kernel.h>  

#include <memory> // std::unique_ptr

__global__ void setup_kernel(curandState *state, const size_t L); 

__global__ void setup_kernel(curandStatePhilox4_32_10_t *state, const size_t L) ;

__global__ void setup_kernel(curandStateMRG32k3a *state, const size_t L) ;


/* =============== custom deleters =============== */
// custom deleters as structs   
struct del_curandState_struct { 
	void operator()(curandState* devStates) { cudaFree(devStates); } 
}; 
struct del_curandStateMRG32k3a_struct {
	void operator()(curandStateMRG32k3a *devMRGStates) { cudaFree(devMRGStates); } 
}; 
struct del_curandStatePhilox4_32_10_t_struct {
	void operator()(curandStatePhilox4_32_10_t *devPHILOXState) {cudaFree(devPHILOXState); }
};  



struct devStatesXOR {
	// (data) members
	std::unique_ptr<curandState[],del_curandState_struct> devStates;  

	// default constructor
	devStatesXOR(); 
	
	// constructors
	devStatesXOR(const size_t) ;
	devStatesXOR(const size_t, const unsigned int, const unsigned int) ;
	
	// move constructor; necessitated by unique_ptr
	devStatesXOR(devStatesXOR &&) ;  
 
	// operator overload assignment = 
	devStatesXOR &operator=(devStatesXOR &&) ;

};

struct devStatesMRG {
	// (data) members
	std::unique_ptr<curandStateMRG32k3a[],del_curandStateMRG32k3a_struct> devStates; 

	// default constructor
	devStatesMRG(); 
	
	// constructors
	devStatesMRG(const size_t) ;
	devStatesMRG(const size_t, const unsigned int, const unsigned int) ;
	
	// move constructor; necessitated by unique_ptr
	devStatesMRG(devStatesMRG &&) ;  
 
	// operator overload assignment = 
	devStatesMRG &operator=(devStatesMRG &&) ;

};
	
struct devStatesPhilox4_32_10_t {
	// (data) members
	std::unique_ptr<curandStatePhilox4_32_10_t[],del_curandStatePhilox4_32_10_t_struct> devStates; 

	// default constructor
	devStatesPhilox4_32_10_t(); 
	
	// constructors
	devStatesPhilox4_32_10_t(const size_t) ;
	devStatesPhilox4_32_10_t(const size_t, const unsigned int, const unsigned int) ;
	
	// move constructor; necessitated by unique_ptr
	devStatesPhilox4_32_10_t(devStatesPhilox4_32_10_t &&) ;  
 
	// operator overload assignment = 
	devStatesPhilox4_32_10_t &operator=(devStatesPhilox4_32_10_t &&) ;

	
};

#endif // END of __XORMRGGENS_H__
