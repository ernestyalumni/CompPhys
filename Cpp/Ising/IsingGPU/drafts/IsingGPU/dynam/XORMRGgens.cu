/**
 * @file   : XORMRGgens.cu
 * @brief  : Examples using cuRAND device API to generate pseudorandom numbers using either XORWOW or MRG32k3a generators, header file   
 * @details : This program uses the device CURAND API.  The purpose of these examples is explore scope and compiling and modularity/separation issues with CURAND   
 * 
 * @author : Ernest Yeung <ernestyalumni@gmail.com>
 * @date   : 20180109      
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
 
#include "./XORMRGgens.h"

__global__ void setup_kernel(curandState *state, const size_t L) 
{
	int id = threadIdx.x + blockIdx.x * blockDim.x; 
	/* Each thread gets same seed, a different sequence 
	 * number, no offset */

	for (int idx = id; idx < L; idx += blockDim.x*gridDim.x) { 
		//curand_init(1234, id, 0, &state[id]); 
		curand_init(1234,idx,0,&state[idx]); 
	}
}

__global__ void setup_kernel(curandStatePhilox4_32_10_t *state, const size_t L) 
{
	int id = threadIdx.x + blockIdx.x * blockDim.x; 
	/* Each thread gets same seed, a different sequence 
	 * number, no offset */ 
	
	for (int idx = id; idx<L; idx+= blockDim.x *gridDim.x) { 
		curand_init(1234, idx, 0, &state[idx]);
	}
}

__global__ void setup_kernel(curandStateMRG32k3a *state, const size_t L) 
{
	int id =threadIdx.x + blockIdx.x * blockDim.x; 
	/* Each thread gets same seed, a different sequence
	 * number, not offset */
	
	for (int idx=id;idx<L;idx+= blockDim.x*gridDim.x) {
		curand_init(1234,idx,0,&state[idx]);
	}
}


// default constructor
devStatesXOR::devStatesXOR() {
	std::unique_ptr<curandState[],del_curandState_struct> devStates_in(nullptr, del_curandState_struct());  
	devStates = std::move(devStates_in);  
	
} 
	
// constructors
devStatesXOR::devStatesXOR(const size_t L) {
	std::unique_ptr<curandState[],del_curandState_struct> devStates_in(nullptr, del_curandState_struct());  
	cudaMallocManaged((void**)&devStates_in,L*sizeof(curandState));
	devStates = std::move(devStates_in);  
	
}	
	
devStatesXOR::devStatesXOR(const size_t L, const unsigned int N_x, const unsigned int M_x) {
	std::unique_ptr<curandState[],del_curandState_struct> devStates_in(nullptr, del_curandState_struct());  
	cudaMallocManaged((void**)&devStates_in,L*sizeof(curandState));
	devStates = std::move(devStates_in);  
	
	setup_kernel<<<N_x,M_x>>>(devStates.get(), L); 
}
	
// move constructor; necessitated by unique_ptr
devStatesXOR::devStatesXOR(devStatesXOR && old_devStatesXOR) :
	devStates { std::move(old_devStatesXOR.devStates) } {} 
 
// operator overload assignment = 
devStatesXOR & devStatesXOR::operator=(devStatesXOR && old_devStatesXOR) {
	devStates = std::move( old_devStatesXOR.devStates ); 
	return *this;
}


// default constructor
devStatesMRG::devStatesMRG() {
	std::unique_ptr<curandStateMRG32k3a[],del_curandStateMRG32k3a_struct> devStates_in(nullptr, del_curandStateMRG32k3a_struct());  
	devStates = std::move(devStates_in);  
	
} 
	
// constructors
devStatesMRG::devStatesMRG(const size_t L) {
	std::unique_ptr<curandStateMRG32k3a[],del_curandStateMRG32k3a_struct> devStates_in(nullptr, del_curandStateMRG32k3a_struct());  
	cudaMallocManaged((void**)&devStates_in,L*sizeof(curandStateMRG32k3a));
	devStates = std::move(devStates_in);  
	
}	
	
devStatesMRG::devStatesMRG(const size_t L, const unsigned int N_x, const unsigned int M_x) {
	std::unique_ptr<curandStateMRG32k3a[],del_curandStateMRG32k3a_struct> devStates_in(nullptr, del_curandStateMRG32k3a_struct());  
	cudaMallocManaged((void**)&devStates_in,L*sizeof(curandStateMRG32k3a));
	devStates = std::move(devStates_in);  
	
	setup_kernel<<<N_x,M_x>>>(devStates.get(), L); 
}
	
// move constructor; necessitated by unique_ptr
devStatesMRG::devStatesMRG(devStatesMRG && old_devStatesMRG) :
	devStates { std::move(old_devStatesMRG.devStates) } {} 
 
// operator overload assignment = 
devStatesMRG & devStatesMRG::operator=(devStatesMRG && old_devStatesMRG) {
	devStates = std::move( old_devStatesMRG.devStates ); 
	return *this;
}



// default constructor
devStatesPhilox4_32_10_t::devStatesPhilox4_32_10_t() {
	std::unique_ptr<curandStatePhilox4_32_10_t[],del_curandStatePhilox4_32_10_t_struct> devStates_in(nullptr, del_curandStatePhilox4_32_10_t_struct());  
	devStates = std::move(devStates_in);  
	
} 
	
// constructors
devStatesPhilox4_32_10_t::devStatesPhilox4_32_10_t(const size_t L) {
	std::unique_ptr<curandStatePhilox4_32_10_t[],del_curandStatePhilox4_32_10_t_struct> devStates_in(nullptr, del_curandStatePhilox4_32_10_t_struct());  
	cudaMallocManaged((void**)&devStates_in,L*sizeof(curandStatePhilox4_32_10_t));
	devStates = std::move(devStates_in);  
	
}	
	
devStatesPhilox4_32_10_t::devStatesPhilox4_32_10_t(const size_t L, const unsigned int N_x, const unsigned int M_x) {
	std::unique_ptr<curandStatePhilox4_32_10_t[],del_curandStatePhilox4_32_10_t_struct> devStates_in(nullptr, del_curandStatePhilox4_32_10_t_struct());  
	cudaMallocManaged((void**)&devStates_in,L*sizeof(curandStatePhilox4_32_10_t));
	devStates = std::move(devStates_in);  
	
	setup_kernel<<<N_x,M_x>>>(devStates.get(), L); 
}
	
// move constructor; necessitated by unique_ptr
devStatesPhilox4_32_10_t::devStatesPhilox4_32_10_t(devStatesPhilox4_32_10_t && old_devStatesPhilox4_32_10_t) :
	devStates { std::move(old_devStatesPhilox4_32_10_t.devStates) } {} 
 
// operator overload assignment = 
devStatesPhilox4_32_10_t & devStatesPhilox4_32_10_t::operator=(devStatesPhilox4_32_10_t && old_devStatesPhilox4_32_10_t) {
	devStates = std::move( old_devStatesPhilox4_32_10_t.devStates ); 
	return *this;
}
