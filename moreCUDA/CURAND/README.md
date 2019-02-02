# on `cuRAND` for `CUDA`  

cf.  [`cuRAND` in CUDA Toolkit Documentation; Introduction](http://docs.nvidia.com/cuda/curand/introduction.html#introduction)

`/include/curand.h` - library on host (CPU) side; random numbers can be generated on device or on host CPU.  
* for device generation, calls to library happen on host, but actual work of random number generation occurs on device  
* resulting random numbers stored in global memory on device  
* Users can then call their own kernels to use random numbers, or copy random numbers back to host for further processing  

`/include/curand_kernel.h` - device header file, defining device functions for setting up random number generator states, and generating sequences of random numbers  
* user-written kernels may then call device functions defined in header file  
* this allows random numbers to be generated and immediatel consumed by user kernels without requiring random numbers to be written to and then read from global memory  

## on [Pseudorandom Sequences](http://docs.nvidia.com/cuda/curand/device-api-overview.html#pseudorandom-sequences)  

### [Bit generation with XORWOW and MRG32k3a generators](http://docs.nvidia.com/cuda/curand/device-api-overview.html#bit-generation-1)  

Following call to `curand_init()`, `curand()` returns sequence of pseudorandom numbers with period greater than $2^{190}$.  

### [Bit generation with MTGP32 generator](http://docs.nvidia.com/cuda/curand/device-api-overview.html#bit-generation-2)   

MTGP32 generator is adaptation of code developed at [Hiroshima University](http://docs.nvidia.com/cuda/curand/bibliography.html#bibliography__saito).  
* samples generated for multiple sequences, each sequence based on set of computed parameters  
	- cuRAND uses 200 parameter sets that have been pre-generated for 32-bit generator with period $2^{11214}$ (!!!)  
* there is 1 state structure for each parameter set (sequence), and 
* algorithm allows thread-safe generation and state update for up to 256 concurrent threads (within single block) for each of the 200 sequences  
* note 2 different blocks can't operate on same state safely.  Also note that, within a block, at most 256 threads may operate on given state.  

For MTGP32 generator, 2 host functions are provided to help set up parameters for different sequences in device memory, and to set up initial state.  

```  
__host__ curandStateust curandMakeMTG32Constants(
	mtgp32paramsfastt params[],
	mtgp32kernelparamst *p)  
```   

```  
__host__ curandStatus_t 
curandMakeMTGP32KernelState(
	curandStateMtgp32_t *s, 
	mtg32_params_fast_t params[], 
	mtgp32_kernel_params_t *k,
	int n, 
	unsigned long long seed)  
```  

and 2 kernel functions  

```  
__device__ unsigned int 
curand (curandStateMtgp32_t *state)  
```  

```  
__device__ unsigned int 
curandmtgp32specific(curandStateMtgp32_t *state, unsigned char index, 
						unsigned char n)  
```  

### [Bit generation with Philox_4x32_10 generator](http://docs.nvidia.com/cuda/curand/device-api-overview.html#bit-generation-3)  

```  
__device__ void 
curand_init (
	unsigned long long seed, unsigned long long subsequence, 
	unsigned long long offset, curandState_t *state)  
```  

```  
__device__ unsigned int 
curand (curandState_t *state)  
```  
Following call to `curand_init()`, `curand()` returns sequence of pseudorandom numbers with period $2^{128}$.  


### [Performance notes](http://docs.nvidia.com/cuda/curand/device-api-overview.html#performance-notes) with the initial state generator; reuse state generator from global memory between kernel launches  	

Calls to `curand_init()` slower than calls to `curand()` or `curand_uniform()`.  It's much faster to save and restore random generator state, e.g. in global memory between kernel launches, used in local memory for fast generation, and stored back into global memory, than recalculate starting state repeatedly.   

```  
__global__ void example(curandState *global_state)  
{
	curandState local_state;
	local_state = global_state[threadIdx.x];
	for (int i=0; i< 10000; i++) {
		unsigned int x = curand(&local_state);
		...
	}
	global_state[threadIdx.x] = local_state; 
}  
```  

### [Device API Examples](http://docs.nvidia.com/cuda/curand/device-api-overview.html#device-api-example)  

cf. [Low Level Bit Hacks You Absolutely Must Know](http://www.catonmat.net/blog/low-level-bit-hacks-you-absolutely-must-know/)  
cf. `./XORMRGgens2distri.cu`  

#### Bit Hack #1. Check if integer is even or odd  
```  
if ((x & 1) == 0 ) {
	x is even  
}
else {
	x is odd 
}
```  
Idea here is integer is odd if and only if least significant bit *b0* is 1, follows from binary represetation of `x`, where bit *b0* contributes to either 1 or 0 - by AND-ing `x` with `1`, eliminate all other bits than *b0*, e.g.  
take integer 43, odd, in binary 43 is 0010101**1**, least significant bit *b0* is **1** (in bold)
```   
  00101011
& 00000001 (note: 1 is same as 00000001) 
  --------
  00000001
```  

So low bit set seems to be odd.  So we'd expect half of the numbers to be odd, the other half to be even.  

## [Modules, Device API](http://docs.nvidia.com/cuda/curand/group__DEVICE.html#group__DEVICE)  

### `curand`

```  
__device__ unsigned int curand ( curandStateXORWOW_t* state)  
```  
Return 32-bits of pseudorandomness from an XORWOW generator.  


