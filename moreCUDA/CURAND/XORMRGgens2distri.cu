/**
 * @file   : XORMRGgens2distri.cu
 * @brief  : Example using cuRAND device API to generate pseudorandom numbers using either XORWOW or MRG32k3a generators   
 * @details : This program uses the device CURAND API to calculate what 
 * proportion of pseudo-random ints have low bit set.  
 * It then generates uniform results to calculate how many 
 * are greater than .5.  
 * It then generates normal results to calculate how many 
 * are within 1 standard deviation of the mean.  
 * 
 * use flags in command-line -m for MRG generator, -p for PHILOX generator  
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
 * nvcc -lcurand XORMRGgens2distri.cu -o XORMRGgens2distri  
 * nvcc -g -lcurand XORMRGgens2distri.cu -o XORMRGgens2distri
 * -g generate debug information for host code, 
 * */
#include <stdio.h>
#include <curand_kernel.h>  

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
	printf("Error at %s:%d\n",__FILE__,__LINE__); \
	return EXIT_FAILURE;}} while(0)  
	
__global__ void setup_kernel(curandState *state) 
{
	int id = threadIdx.x + blockIdx.x * 64; 
	/* Each thread gets same seed, a different sequence 
	 * number, no offset */ 
	curand_init(1234, id, 0, &state[id]); 
}  

__global__ void setup_kernel(curandStatePhilox4_32_10_t *state) 
{
	int id = threadIdx.x + blockIdx.x * 64; 
	/* Each thread gets same seed, a different sequence 
	 * number, no offset */ 
	curand_init(1234, id, 0, &state[id]) ;
}

__global__ void setup_kernel(curandStateMRG32k3a *state) 
{
	int id = threadIdx.x + blockIdx.x * 64; 
	/* Each thread gets same seed, a different sequence 
	 * number, no offset */ 
	curand_init(0, id, 0, &state[id]);
}

__global__ void generate_kernel(curandState *state, 
								int n,
								unsigned int *result)
{
	int id = threadIdx.x + blockIdx.x * 64; 
	int count = 0;
	unsigned int x; 
	/* Copy state to local memory for efficienty */ 
	curandState localState = state[id]; 
	/* Generate pseudo-random unsigned ints */
	for (int i=0; i<n; i++) {
		x = curand(&localState); 
		/* Check if low bit set */
		if (x & 1) {
			count++;
		}
	} 
	/* Copy state back to global memory */ 
	state[id] = localState; 
	/* Store results */
	result[id] += count;
}

__global__ void generate_kernel(curandStatePhilox4_32_10_t *state, 
								int n, unsigned int *result) 
{
	int id = threadIdx.x + blockIdx.x * 64; 
	int count = 0;
	unsigned int x;
	/* Copy state to local memory for efficiency */
	curandStatePhilox4_32_10_t localState = state[id]; 
	/* Generate pseudo-random unsigned ints */
	for (int i=0; i<n;i++) {
		x=curand(&localState); 
		/* Check if low bit set */
		if (x & 1) {
			count++; 
		}
	}
	/* Copy state back to global memory */
	state[id] = localState;
	/* Store results */
	result[id] += count; 
}  

__global__ void generate_uniform_kernel(curandState *state,
											int n,
											unsigned int *result) 
{
	int id = threadIdx.x + blockIdx.x * 64;  
	unsigned int count = 0;
	float x;
	/* Copy state to local memory for efficiency */ 
	curandState localState = state[id]; 
	/* Generate pseudo-random uniforms */
	for (int i=0; i<n; i++) {
		x = curand_uniform(&localState);
		/* Check if > .5 */
		if (x > .5) {
			count++; 
		}
	}
	/* Copy state back to global memory */
	state[id] = localState; 
	/* Store results */
	result[id] += count; 
}

__global__ void generate_uniform_kernel(curandStatePhilox4_32_10_t *state,
										int n,
										unsigned int *result) 
{
	int id = threadIdx.x + blockIdx.x * 64; 
	unsigned int count = 0;
	float x;
	/* Copy state to local memory for efficiency */ 
	curandStatePhilox4_32_10_t localState = state[id];  
	/* Generate pseudo-random uniforms */
	for (int i=0; i<n; i++) {
		x = curand_uniform(&localState);
		/* Check if > .5 */
		if (x > .5) {
			count++; 
		}
	}
	/* Copy state back to global memory */
	state[id] = localState; 
	/* Store results */
	result[id] += count;
}  

__global__ void generate_normal_kernel(curandState *state, 
										int n, 
										unsigned int *result)  
{ 
	int id = threadIdx.x + blockIdx.x * 64; 
	unsigned int count = 0;
	float2 x;
	/* Copy state to local memory for efficiency */ 
	curandState localState = state[id]; 
	/* Generate pseudo-random normals */
	for (int i=0; i<n/2; i++) {
		x = curand_normal2(&localState);
		/* Check if within 1 standard deviation */
		if ((x.x > -1.0) && (x.x < 1.0)) {
			count++;
		} 
		if ((x.y > -1.0) && (x.y < 1.0)) {
			count++;
		}
	}
	/* Copy state back to global memory */
	state[id] = localState;
	/* Store results */
	result[id] += count;
}

__global__ void generate_normal_kernel(curandStatePhilox4_32_10_t *state, 
										int n,
										unsigned int *result) 
{
	int id = threadIdx.x + blockIdx.x * 64; 
	unsigned int count = 0;
	float2 x;
	/* Copy state to local memory for efficiency */ 
	curandStatePhilox4_32_10_t localState = state[id]; 
	/* Generate pseudo-random normals */ 
	for (int i=0; i<n/2; i++) {
		x = curand_normal2(&localState); 
		/* Check if within 1 standard deviation */
		if ((x.x > -1.0) && (x.x < 1.0)) {
			count++;
		}
		if ((x.y > -1.0) && (x.y < 1.0)) {
			count++; 
		}
	}
	/* Copy state back to global memory */
	state[id]= localState;
	/* Store results */
	result[id] += count; 
}

__global__ void generate_kernel(curandStateMRG32k3a *state,
								int n,
								unsigned int *result) 
{
	int id = threadIdx.x + blockIdx.x * 64;
	unsigned int count = 0;
	unsigned int x; 
	/* Copy state to local memory for efficiency */
	curandStateMRG32k3a localState = state[id]; 
	/* Generate pseudo-random unsigned ints */
	for (int i=0; i<n; i++) {
		x = curand(&localState); 
		/* Check if low bit set */
		if (x & 1) {
			count++; 
		} 
	}
	/* Copy state back to global memory */ 
	state[id] = localState; 
	/* Store results */ 
	result[id] += count;
}

__global__ void generate_uniform_kernel(curandStateMRG32k3a *state, 
										int n,
										unsigned int *result) 
{
	int id = threadIdx.x + blockIdx.x * 64;  
	unsigned int count = 0;
	double x;
	/* Copy state to local memory for efficiency */
	curandStateMRG32k3a localState = state[id]; 
	/* Generate pseudo-random uniforms */
	for (int i=0; i<n; i++) {
		x= curand_uniform_double(&localState);
		/* Check if > .5 */
		if (x>.5) {
			count++; 
		} 
	}
	/* Copy state back to global memory */ 
	state[id] = localState;
	/* Store results */
	result[id] += count;
} 

__global__ void generate_normal_kernel(curandStateMRG32k3a *state,
										int n, 
										unsigned int *result)
{
	int id = threadIdx.x + blockIdx.x * 64;
	unsigned int count =0;
	double2 x;
	/* Copy state to local memory for efficiency */
	curandStateMRG32k3a localState = state[id]; 
	/* Generate pseudo-random normals */
	for (int i=0; i <n/2; i++) {
		x = curand_normal2_double(&localState);
		/* Check if within 1 standard deviation */
		if ((x.x > -1.0) && (x.x <1.0)) {
			count++; 
		}
		if((x.y > -1.0) && (x.y < 1.0)) {
			count++; 
		}
	}
	/* Copy state back to global memory */
	state[id] = localState; 
	/* Store results */
	result[id] += count;
}

int main(int argc, char *argv[]) 
{
	int i; 
	unsigned int total;
	curandState *devStates;
	curandStateMRG32k3a *devMRGStates;
	curandStatePhilox4_32_10_t *devPHILOXStates; 
	unsigned int *devResults, *hostResults; 
	bool useMRG = 0;
	bool usePHILOX = 0;
	int sampleCount = 10000;
	bool doubleSupported = 0;
	int device;
	struct cudaDeviceProp properties; 
	
	/* check for double precision support */
	CUDA_CALL(cudaGetDevice(&device));
	CUDA_CALL(cudaGetDeviceProperties(&properties,device));
	if (properties.major >= 2 || (properties.major == 1 && properties.minor >= 3) ) {
		doubleSupported = 1; 
	}
	
	/* Check for MRG32k3a option (default is XORWOW) */ 
	if (argc >= 2) {
		if (strcmp(argv[1], "-m") == 0) {
			useMRG = 1; 
			if (!doubleSupported) {
				printf("MRG32k3a requires double precision\n"); 
				printf("^^^^ test WAIVED due to lack of double precision\n"); 
				return EXIT_SUCCESS; 
			}
		} else if (strcmp(argv[1],"-p") ==0) {
				usePHILOX = 1;
		} 
		/* Allow over-ride of sample count */
		sscanf(argv[argc-1], "%d", &sampleCount);
	}

	/* Allocate space for results on host */
	hostResults = (unsigned int *)calloc(64 * 64, sizeof(int)); 
	
	/* Allocate space for results on device */
	CUDA_CALL(cudaMalloc((void **)&devResults, 64 * 64 * 
				sizeof(unsigned int)));  
				
	/* Set results to 0 */
	CUDA_CALL(cudaMemset(devResults, 0 ,64*64 * 
				sizeof(unsigned int)));
				
	/* Allocate space for prng states on device; prng=Pseudorandom Number Generator */
	if (useMRG) {
		CUDA_CALL(cudaMalloc((void**)&devMRGStates, 64*64*
					sizeof(curandStateMRG32k3a)));
	} else if (usePHILOX) {
		CUDA_CALL(cudaMalloc((void**)&devPHILOXStates, 64*64 *
					sizeof(curandStatePhilox4_32_10_t)));
	} else {
		CUDA_CALL(cudaMalloc((void **)&devStates, 64*64 *
					sizeof(curandState)));
	}  
	
	/* Setup prng states */
	if (useMRG) {
		setup_kernel<<<64, 64>>>(devMRGStates); 
	} else if (usePHILOX) 
	{
		setup_kernel<<<64,64>>>(devPHILOXStates);
	} else {
		setup_kernel<<<64, 64>>>(devStates); 
	}
	
	/* Generate and use pseudo-random */
	for (i=0; i < 50; i++) {
		if (useMRG) {
			generate_kernel<<<64,64>>>(devMRGStates, sampleCount, devResults); 
		} else if (usePHILOX) {
			generate_kernel<<<64,64>>>(devPHILOXStates, sampleCount, devResults); 
		} else {
			generate_kernel<<<64,64>>>(devStates, sampleCount, devResults); 
		}
	}
	
	/* Copy device memory to host */
	CUDA_CALL(cudaMemcpy(hostResults, devResults, 64 * 64 * 
				sizeof(unsigned int), cudaMemcpyDeviceToHost)); 
				
	/* Show results */
	total = 0;
	for (i=0 ; i < 64*64; i++) {
		total+= hostResults[i]; 
	} 
	printf("Fraction with low bit set was %10.13f\n", 
		(float)total / (64.0f * 64.0f * sampleCount * 50.0f)); 
		
	/* Set results to 0 */ 
	CUDA_CALL(cudaMemset(devResults, 0, 64*64*sizeof(unsigned int)));
	
	/* Generate and use uniform pseudo-random */ 
	for (i=0; i< 50; i++) {
		if (useMRG) {
			generate_uniform_kernel<<<64, 64>>>(devMRGStates, sampleCount, devResults); 
		} else if (usePHILOX) {
			generate_uniform_kernel<<<64, 64>>>(devPHILOXStates, sampleCount, devResults); 
		} else {
			generate_uniform_kernel<<<64, 64 >>>(devStates, sampleCount, devResults);  
		}
	} 
	
	/* Copy device memory to host */
	CUDA_CALL(cudaMemcpy(hostResults, devResults, 64 * 64 * 
		sizeof(unsigned int), cudaMemcpyDeviceToHost)); 
	
	/* Show result */
	total = 0;
	for (i=0; i < 64 * 64 ; i++) {
		total += hostResults[i]; 
	}
	printf("Fraction of uniforms > 0.5 was %10.13f\n", 
		(float) total / (64.0f * 64.0f * sampleCount * 50.0f)); 
	
	/* Set results to 0 */
	CUDA_CALL(cudaMemset(devResults, 0, 64 * 64 * 
				sizeof(unsigned int)));  
				
	/* Generate and use normal pseudo-random */
	for (i=0; i < 50; i++) {
		if (useMRG) {
			generate_normal_kernel<<<64, 64>>>(devMRGStates, sampleCount, devResults); 
		} else if (usePHILOX) {
			generate_normal_kernel<<<64, 64>>>(devPHILOXStates, sampleCount, devResults);  
		} else {
			generate_normal_kernel<<<64, 64>>>(devStates, sampleCount, devResults); 
		}
	}
	
	
	/* Copy device memory to host */
	CUDA_CALL(cudaMemcpy(hostResults, devResults, 64* 64*
		sizeof(unsigned int), cudaMemcpyDeviceToHost));  
		
	/* Show result */ 
	total = 0;
	for (i=0; i< 64*64; i++) {
		total += hostResults[i]; 
	} 
	printf("Fraction of normals within 1 standard deviation was %10.13f\n", 
		(float) total/(64.0f * 64.0f * sampleCount *50.0f)); 
	
	/* Cleanup */
	if (useMRG) {
		CUDA_CALL(cudaFree(devMRGStates));
	} else if (usePHILOX) 
	{
		CUDA_CALL(cudaFree(devPHILOXStates));
	} else {
		CUDA_CALL(cudaFree(devStates));
	}
	
	
	CUDA_CALL(cudaFree(devResults));
	free(hostResults); 
	printf("^^^^ kernel_example PASSED\n");
	return EXIT_SUCCESS;	
}

	
	

