/**
 * @file   : main.cpp
 * @brief  : main driver file for Examples using cuRAND device API to generate pseudorandom numbers using either XORWOW or MRG32k3a generators, header file   
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
#include "./gens2distri/XORMRGgens.h"  

int main(int argc, char* argv[]) 
{ 
	/* ========== grid, thread dims. ========== */
	constexpr const unsigned int M_x = 128; // number of threads, per block, in x-direction 
	constexpr const unsigned int N_x = 128; // number of (thread) blocks on the grid, in x-direction  

	// custom deleters for curandStates's in main (so scope is in main)   
	// custom deleter as a lambda function  
	auto del_curandState_lambda_main=[&](curandState* devStates) {cudaFree(devStates);};
	auto del_curandStateMRG32k3a_lambda_main=[&](curandStateMRG32k3a *devMRGStates) { cudaFree(devMRGStates); }; 
	auto del_curandStatePhilox4_32_10_t_lambda_main=[&](curandStatePhilox4_32_10_t *devPHILOXStates) {cudaFree(devPHILOXStates); };
	
	// custom deleters as a STRUCT 
	struct del_curandState_struct_main { 
		void operator()(curandState* devStates) { cudaFree(devStates); } 
	}; 
	struct del_curandStateMRG32k3a_struct_main {
		void operator()(curandStateMRG32k3a *devMRGStates) { cudaFree(devMRGStates); } 
	}; 
	struct del_curandStatePhilox4_32_10_t_struct_main {
		void operator()(curandStatePhilox4_32_10_t *devPHILOXState) {cudaFree(devPHILOXState); }
	};  
	
	// unique_ptrs for curandStates's 
	std::unique_ptr<curandState,decltype(del_curandState_lambda_main)> devStates_lambda_main(nullptr, del_curandState_lambda_main); 
	std::unique_ptr<curandState,del_curandState_struct_main> devStates_struct_main(nullptr, del_curandState_struct_main());  
	
	std::unique_ptr<curandStateMRG32k3a,decltype(del_curandStateMRG32k3a_lambda_main)> devMRGStates_lambda_main(nullptr, del_curandStateMRG32k3a_lambda_main); 
	std::unique_ptr<curandStateMRG32k3a,del_curandStateMRG32k3a_struct_main> devMRGStates_struct_main(nullptr, del_curandStateMRG32k3a_struct_main());  
	
	std::unique_ptr<curandStatePhilox4_32_10_t,decltype(del_curandStatePhilox4_32_10_t_lambda_main)> devPHILOXStates_lambda_main(nullptr, del_curandStatePhilox4_32_10_t_lambda_main); 
	std::unique_ptr<curandStatePhilox4_32_10_t,del_curandStatePhilox4_32_10_t_struct_main> devPHILOXStates_struct_main(nullptr, del_curandStatePhilox4_32_10_t_struct_main());
	
	// shared_ptrs for curandStates's  
	
	
}
