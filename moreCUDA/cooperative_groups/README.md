# on Cooperative Groups    

cf. [Cooperative Groups: Flexible CUDA Thread Programming | Parallel Forall](https://devblogs.nvidia.com/parallelforall/cooperative-groups/)  


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
