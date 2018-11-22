/**
 * @file   : main.cu
 * @brief  : main driver file for Examples using cuRAND device API to generate pseudorandom numbers using either XORWOW or MRG32k3a generators, header file   
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
 * nvcc -lcurand main.cu ./gens2distri/XORMRGgens.cu -o main  
 * */

#include "./gens2distri/XORMRGgens.h" // 

#include <iostream>

/* ********** functions to setup device GPU ********** */

/** @fn getMaxGridSize
 * @brief get maxGridSize (total number threads on a (thread) grid, on device GPU, of a single device GPU
 * */
size_t get_maxGridSize() {
	cudaDeviceProp prop;
	int count;
	cudaGetDeviceCount(&count);
	size_t MAXGRIDSIZE; 
	if (count>0) {
		cudaGetDeviceProperties(&prop, 0);
		MAXGRIDSIZE = prop.maxGridSize[0]; 
		return MAXGRIDSIZE; 
	} else { return EXIT_FAILURE; }
}; 

/** @fn generate_kernel
 * @param n - for each thread, generate n random unsigned ints; 1 reason to do this is to utilize the compute of a thread 
 * */
__global__ void generate_kernel(curandState *state, 
								int n, 
								unsigned int *result, const unsigned long int L) 
{
	int id = threadIdx.x + blockIdx.x * blockDim.x ; 

	for (int idx = id; idx < L; idx += blockDim.x * gridDim.x ) { 
		unsigned int count = 0; 
		unsigned int x; // 
	
		/* Copy state to local memory for efficiency */ 
		curandState localState = state[idx]; 
		/* Generate pseudo-random unsigned ints */ 
		for (int i=0; i<n; i++) {
			x = curand(&localState); 
			/* Check if low bit set */ // i.e. if it's odd or not 
			if (x & 1) { 
				count++; 
			}
		}
		/* Copy state back to global memory */
		state[idx] = localState; 
		result[idx] += count; 
	}	
}

/** @fn generate_kernel
 * @param n - for each thread, generate n random unsigned ints; 1 reason to do this is to utilize the compute of a thread 
 * */
__global__ void generate_kernel(curandStatePhilox4_32_10_t *state, 
								int n, 
								unsigned int *result, const unsigned long int L) 
{
	int id = threadIdx.x + blockIdx.x * blockDim.x ; 

	for (int idx = id; idx < L; idx += blockDim.x * gridDim.x ) { 
		
		unsigned int count = 0; 
		unsigned int x; // 

		/* Copy state to local memory for efficiency */ 
		curandStatePhilox4_32_10_t localState = state[idx]; 
		/* Generate pseudo-random unsigned ints */ 
		for (int i=0; i<n; i++) {
			x = curand(&localState); 
			/* Check if low bit set */ // i.e. if it's odd or not 
			if (x & 1) { 
				count++; 
			}
		}
		/* Copy state back to global memory */
		state[idx] = localState; 
		result[idx] += count; 
	}	
}

/** @fn generate_uniform_kernel
 * @param n - for each thread, generate n random unsigned ints; 1 reason to do this is to utilize the compute of a thread 
 * */
__global__ void generate_uniform_kernel(curandState *state, 
								int n, 
								unsigned int *result, const unsigned long int L) 
{
	int id = threadIdx.x + blockIdx.x * blockDim.x ; 

	for (int idx = id; idx < L; idx += blockDim.x * gridDim.x ) { 
		unsigned int count = 0; 
		float x; 
	
		/* Copy state to local memory for efficiency */ 
		curandState localState = state[idx]; 
		/* Generate pseudo-random unsigned ints */ 
		for (int i=0; i<n; i++) {
			x = curand_uniform(&localState); 
			/* Check if > .5 */  
			if (x > .5) { 
				count++; 
			}
		}
		/* Copy state back to global memory */
		state[idx] = localState; 
		result[idx] += count; 
	}	
}

/** @fn generate_uniform_kernel
 * @param n - for each thread, generate n random unsigned ints; 1 reason to do this is to utilize the compute of a thread 
 * */
__global__ void generate_uniform_kernel(curandStatePhilox4_32_10_t *state, 
								int n, 
								unsigned int *result, const unsigned long int L) 
{
	int id = threadIdx.x + blockIdx.x * blockDim.x ; 
	unsigned int count = 0; 
	float x; 

	for (int idx = id; idx < L; idx += blockDim.x * gridDim.x ) { 
		/* Copy state to local memory for efficiency */ 
		curandStatePhilox4_32_10_t localState = state[idx]; 
		/* Generate pseudo-random unsigned ints */ 
		for (int i=0; i<n; i++) {
			x = curand_uniform(&localState); 
			/* Check if > .5 */  
			if (x > .5) { 
				count++; 
			}
		}
		/* Copy state back to global memory */
		state[idx] = localState; 
		result[idx] += count; 
	}	
}


 
int main(int argc, char* argv[]) 
{
	/* ***** (thread) grid,block dims ***** */ 
	/* min of N_x, number of (thread) blocks on grid in x-direction, and MAX_BLOCKS allowed is 
	 * determined here */
	size_t MAXGRIDSIZE = get_maxGridSize();  
	unsigned int M_x = 1<<8;  // M_x = number of threads in x-direction, in a single block, i.e. blocksize; 2^8=256  
	unsigned int L = 1<<18; // doesn't output correct values for n = 1<<39    
	unsigned int MAX_BLOCKS = (MAXGRIDSIZE + M_x - 1)/ M_x; 
	// notice how we're only launching 1/4 of L threads
	unsigned int N_x = min( MAX_BLOCKS, ((L + M_x - 1)/ M_x)); 
	/* ***** END of (thread) grid,block dims ***** */ 


	// Use structs devStatesXOR, devStatesMRG, devStatesPhilox4_32_10_t to automate process of setting up curandStates  
	devStatesXOR devstatesXOR = { L, N_x, M_x } ;
	devStatesMRG devstatesMRG = { L, N_x, M_x } ;
	devStatesPhilox4_32_10_t devstatesPhilox4_32_10_t = { L, N_x, M_x } ;


	// set the sampleCount
	constexpr const int sampleCount = 10000;

	/* Allocate space for results on host */ 
	auto hostResults = std::make_unique<unsigned int[]>(L);  

	/* Allocate space for results on device */
	// custom deleter for unsigned int array, as a lambda function 
	auto del_devResults_lambda_main=[&](unsigned int* devResults) {cudaFree(devResults); }; 
	std::unique_ptr<unsigned int[],decltype(del_devResults_lambda_main)> devResults(nullptr, del_devResults_lambda_main);  
	cudaMallocManaged((void **)&devResults, L*sizeof(unsigned int));  

	/* Set results to 0 */ 
	cudaMemset(devResults.get(), 0, L*sizeof(unsigned int) );

	/* Generate and use pseudo-random */
	/* this will test if we have low bit set, i.e. odd numbers */   
	for (int i=0; i < 50; i++) {
		generate_kernel<<<N_x,M_x>>>(devstatesXOR.devStates.get(), sampleCount, devResults.get(), L );
	}
	
	/* Copy device memory to host */
	cudaMemcpy(hostResults.get(), devResults.get(), L * sizeof(unsigned int), cudaMemcpyDeviceToHost); 
	
	/* Show results */
	unsigned long long int total = 0;
	for (int i =0; i < L; i++) {
		total += hostResults[i]; 
	}
	std::cout << "Fraction with low bit set was " << (float)total / (L * sampleCount * 50.0f) << std::endl; 
	

	/* Set results to 0 */
	cudaMemset(devResults.get(), 0, L * sizeof(unsigned int));
	
	/* Generate and use uniform pseudo-random */
	for (int i=0; i<50; i++) {
		generate_uniform_kernel<<<N_x,M_x>>>(devstatesXOR.devStates.get(), sampleCount, devResults.get(), L); 
	}
	
	/* Copy device memory to host */
	cudaMemcpy(hostResults.get(), devResults.get(), L * sizeof(unsigned int), cudaMemcpyDeviceToHost); 
	
	/* Show result */
	total =0;
	for (int i=0; i < L; i++) {
		total += hostResults[i]; 
	}
	std::cout << "Fraction of uniforms > 0.5 was " << (float) total / ( (float) L * sampleCount * 50.0f ) << std::endl;


}
