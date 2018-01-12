# on Cooperative Groups    

cf. [Cooperative Groups: Flexible CUDA Thread Programming | Parallel Forall](https://devblogs.nvidia.com/parallelforall/cooperative-groups/)   
cf. [Cooperative Groups: Kyrylo Perelygin, Yuan Lin, GTC 2017](http://on-demand.gputechconf.com/gtc/2017/presentation/s7622-Kyrylo-perelygin-robust-and-scalable-cuda.pdf)
cf. [C. Cooperative Groups, CUDA Toolkit Documentation](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cooperative-groups)  

### Correct way to loop over all the elements in an array of length L, even when L is not a power of 2 or 4, for vector loading    

e.g. `../cooperative_groups/cg_eg3.cu`, `__device__ int thread_sum(int *, int L)`  

```  
	unsigned int k_x = threadIdx.x + blockDim.x*blockIdx.x; 
	
	/* increment by blockDim.x*gridDim.x, so that a single thread will do all the 
	 * "work" needed done on n, especially if n >= gridDim.x*blockDim.x = N_x*M_x */	
	for (int i=k_x; 
			i < L/4; 
			i += blockDim.x * gridDim.x) 
	{
		int4 in = ((int4*) input)[i]; 
		sum += in.x + in.y + in.z + in.w; 
	}
	// process remaining elements
	for (unsigned int idx= k_x + L/4*4; idx < L; idx += 4 ) {
		sum += input[idx];
	}
```  

and launch these number of (thread) blocks, given `M_x` number of threads in a single (thread) block (in x-direction):  

```  
	// notice how we're only launching 1/4 of L threads
	N_x = min( MAX_BLOCKS, ((L/4 + M_x - 1)/ M_x)); 

sum_kernel<<<N_x,M_x,>>>(sum.get(),input.get(),L); 

```

Note the 2 loops:  
```  
	unsigned int k_x = threadIdx.x + blockDim.x*blockIdx.x; 
	
	/* increment by blockDim.x*gridDim.x, so that a single thread will do all the 
	 * "work" needed done on n, especially if n >= gridDim.x*blockDim.x = N_x*M_x */	
	for (int i=k_x; 
			i < L/4; 
			i += blockDim.x * gridDim.x) 
```  
and  

```  
	// process remaining elements
	for (unsigned int idx= k_x + L/4*4; idx < L; idx += 4 ) {
```  

## Partitioning Groups; Tiled Partitions   

cf. ["Partitioning Groups" sec. of Parallel for All blog](https://devblogs.nvidia.com/parallelforall/cooperative-groups/)  
cf. ["Thread Block Tile", slide 9, from GTC 2017 presentation](http://on-demand.gputechconf.com/gtc/2017/presentation/s7622-Kyrylo-perelygin-robust-and-scalable-cuda.pdf)  
cf. ["C.2.2. Tiled Partitions"](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#tiled-partitions-cg)

Error in ["Partitioning Groups" sec. of Parallel for All blog](https://devblogs.nvidia.com/parallelforall/cooperative-groups/): in 
```  
thread_group tile32 = cg::partition(this_thread_block(), 32);  
``` 
I don't think there's a `cg::partition`, but there is a `cg::tiled_partition` and it's probably meant to be the latter (I'm on CUDA 9, GeForce GTX 980 Ti; btw, any hardware donation for a Titan V or GTX 1080 Ti would be welcome!).  Otherwise, I obtain this error, `error: namespace "cooperative_groups" has no member "partition"`.    

`cg_eg4.cu` - this is the full example of using partitioning groups, tiled partitions, partitioning a whole thread block into tiles of 32 threads, and then to 4 tiles, with "driver" functions, and a `__global__` "driver" kernel to demonstrate modularity.  

## Grid Synchronization, instead of `<<<...>>>`, use `cudaLaunchCooperativeKernel`  

cf. [C.3. Grid Synchronization of CUDA Toolkit Documentation v.9](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#grid-synchronization-cg)  

```  
cudaLaunchCooperativeKernel(
	const T *func, 
	dim3 gridDim,
	dim3 blockDim,
	void **args, 
	size_t sharedMem =0,
	cudaStream_t stream =0
)
```  

cf. [cudaLaunchCooperativeKernel](http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html#group__CUDART__EXECUTION_1g504b94170f83285c71031be6d5d15f73)


### Short note on number of (thread) blocks to launch, N_x (in the x-direction); how to get device GPU's max. blocks and compare to how many blocks you need  

Get the `.maxGridsize` of a single device GPU inside of `main` function or as a function:  
```  
cudaDeviceProp prop;
int count;
cudaGetDeviceCount(&count); 
int MAXGRIDSIZE;
if (count >0) {
	cudaGetDeviceProperties( &prop, 0); 
	MAXGRIDSIZE = prop.maxGridSize[0];
} else { return EXIT_FAILURE; }
```  
or as a function (call) `get_maxGridSize()`  
```  
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
``` 
On my nVidia GeForce GTX 980Ti (I need a hardware donation; please donate a TITAN V or GTX 1080Ti if you find my work useful!), `.maxGridSize[0]` (in x-direction) is   
```  
2147483647
```  
which is 2^31, which should be the theoretical max. value that can be stored in a 32-bit unsigned int.  What if you're doing multi-GPUs, with more than 2^(31) threads available?  Then I would think you'd need `size_t` to store this value, not `unsigned int`.  

Then do this formula:  
```  
N_x = (L_x + M_x - 1)/M_x; 
```  
where `M_x` is the number of threads in a single (thread) block, and `L` is either 2 possibilities:  
- the total size of the array you have, `L`, or     
- max. number of threads allowed on the device GPU, that you found from, say, `get_maxGridSize()`, `MAXGRIDSIZE`.  
and then you calculate `N_x_needed`, or `N_x_MAX`, respectively.  

Clearly, you want to take the `min` of `N_x_needed` and `N_x_MAX` (because otherwise, you'd launch more threads than physically allowed on device GPU hardware).  
However, it is a question of whether you want to do this check in the `main` driver function, or inside a driver function.  If the latter, do this, for example:  
```  
void device_copy_vector4(int* d_in, int* d_out, int N, unsigned int MAX_BLOCKS) {
	int threads = 128; 
	int blocks=min((N/4+threads-1)/ threads, MAX_BLOCKS); 
	
	device_copy_vector4_kernel<<<blocks,threads>>>(d_in,d_out,N); 
}
```  




