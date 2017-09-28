# Bringing CUDA into the year 2011: C++11 smart pointers with CUDA, CUB, `nccl`, streams, and CUDA Unified Memory Management with CUB and CUBLAS     

Ernest Yeung <ernestyalumni@gmail.com>

## Summary  

First, I was motivated by the need to load large arrays onto device GPU global memory, sometimes from batches of CPU host memory for machine learning/deep learning applications.  This could also be necessitated by the bottleneck of having only so much data available externally, while GPU utilization is optimized for large device GPU arrays.  

*I show how this can be resolved with C++11 smart pointers.*  Usage of these C++11 smart pointers, not only being the latest, best practices, *automates the freeing* up of memory and provides a *safe* way to point to the raw pointer when needed.    

I also show how to use *CUDA Unified Memory Management* to automate memory transfers between CPU to GPU and multi-GPUs.  I show its use with *CUB* (for parallel `reduce` and `scan` algorithms), and CUBLAS, for linear algebra.  

Also for CUB, `nccl` (parallel `reduce` and `scan` for multi-GPUs), CUDA streams, I show how to wrap device GPU arrays with C++11 smart pointers, to, again, automate the freeing up of memory and provide a safe way to point to the raw pointer when needed.    

While I've seen and have only been able to encounter a great amount of CUDA code written in CUDA *C*, I've sought to show best practices in using CUDA C++11, setting up the stage for the next best practices standards, when CUDA will use C++17.  

## A brief recap of CUDA Unified Memory Management  

The salient and amazing feature of CUDA Unified Memory Management is that CUDA is automating how to address the memory to be allocated for the desired array of data both on the CPU and the GPU.  *This is especially useful for multi-GPU setups.*  You *don't* want to manually address memory on a number of GPUs.  

### Motivation; before CUDA Unified Memory Management, before CUDA 6  

Before CUDA Unified Memory Management, before CUDA 6, one *had* to allocate, separately, 2 arrays, 1 on the host, and another, of exact, same size, on the device.  

For example (cf. [`withoutunifiedmem.cu`](https://raw.githubusercontent.com/ernestyalumni/CompPhys/mobile/moreCUDA/CUDACpp11/withoutunifiedmem.cu)),  
```  
# host array
int *host_ret = (int *)malloc(1000 * sizeof(int));  

# device array
int *ret;  
cudaMalloc(&ret, 1000 * sizeof(int)); 

# after computation on GPU, it would be useful to leave the result on the GPU; we have to get reuslt out to the user   
cudaMemcpy(host_ret, ret, 1000*sizeof(int), cudaMemcpyDeviceToHost);

free(host_ret);
cudaFree(ret);  
```  
Note that one needs to allocate (and free!) 2 separate arrays, host and device, and then `cudaMemcpy` between host and device - and CPU-GPU memory transfers are (relatively) slow!  

### With CUDA Unified Memory Management; `cudaMallocManaged`  

With CUDA Unified Memory, allocate (and destroy) only *1* array with `cudaMallocManaged` (cf. [`unifiedmem.cu`](https://github.com/ernestyalumni/CompPhys/blob/mobile/moreCUDA/CUDACpp11/unifiedmem.cu)):  

```  
int *ret;
cudaMallocManaged(&ret, 1000*sizeof(int));
AplusB<<<1,1000>>>(ret, 10,100);

/*
 * In non-managed example, synchronous cudaMemcpy() routine is used both 
 * to synchronize the kernel (i.e. wait for it to finish running), &
 * transfer data to host.  
 * The Unified Memory examples do not call cudaMemcpy() and so 
 * require an explicit cudaDeviceSynchronize() before host program
 * can safely use output from GPU.  
 */
cudaDeviceSynchronize();
	for (int i=0; i<1000; i++) {
		printf("%d: A+B = %d\n", i,ret[i]);
	}
cudaFree(ret);  
```  
It is very important that you now have to be considerate of synchronizing of a kernel run on the GPU with GPU-CPU data transfers, as mentioned above in the code.  Thus, `cudaDeviceSynchronize` was inserted in between the example kernel (run on the GPU) `AplusB` and the printing of the array on the host CPU (`printf`). 

See also [`unifiedcoherency.cu`](https://github.com/ernestyalumni/CompPhys/blob/mobile/moreCUDA/CUDACpp11/unifiedcoherency.cu)  

For completeness, one can also declare globally ("at the top of your code") 

```  
__device__ __managed__ int ret[1000]; 
```    

cf. [`unifiedmem_direct.cu`](https://github.com/ernestyalumni/CompPhys/blob/mobile/moreCUDA/CUDACpp11/unifiedmem_direct.cu)  

However, I've found that, unless, an array is specifically needed to have global scope, such as with OpenGL interoperability, it's unwielding to hardcode a specific array for global scope ("at the top of the code").  





Again, for completeness, I will briefly describe `cudaMallocHost`.  

`cudaMallocHost` allows for the allocation of *page-locked* memory on the host - meaning *pinned* memory; the memory is allocated "firmly" or its address is fixed on the host so that CUDA knows where it exactly is, and can automatically optimize CPU-GPU data transfers between this fixed host memory and device GPU memory (remember, CUDA cannot directly access host CPU memory!).  

A full, working example is [here, `cudaMallocHost_eg.cu`](https://github.com/ernestyalumni/CompPhys/blob/mobile/moreCUDA/CUDACpp11/cudaMallocHost_eg.cu), but the gist of the creation (and important destruction) of a cudaMallocHost'ed array is here:  

```  
float *a; 
		
cudaMallocHost(&a, N_0*sizeof(float));

cudaFreeHost(a);
```  

cf. [4.9 Memory Management, CUDA Runtime API, CUDA Toolkit v9.0.176 Documentation](http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1gab84100ae1fa1b12eaca660207ef585b)  


## Doing (C++11) smart pointer arithmetic directly on device GPU memory so to load "batches" of data from the host onto portions of the device GPU array! (!!!)  

This is one of the milestones of this discussion.  

I was concerned deeply with the transfer of data on the CPU (RAM) to the device GPU memory in the application to running *deep learning models*.  

In practice, the bottlenecks are the slow transfer of data between the CPU and GPU.  Second of all, to optimize the utilization of the GPU, one should launch as many threads as possible (e.g. 65536 total allowed threads for the "Max. grid dimensions" on this GTX 1050), and, roughly speak, each of those threads should have as much data to work with as possible on GPU global memory.

*As much data to be processed should be loaded from the CPU and onto a device GPU array as possible, and the device GPU array should be as large as possible so to provide all those threads with stuff to compute.*   

In fact, suppose the goal is to load an array of length (i.e. number of elements) `Lx` onto the device GPU global memory.  

Suppose we can only load it in "batches", say `n=2` batches.  Some information from the outside may come, sequentially (in time), before the other.  

In this simple (but instructive) `n=2` case, say we have data for the first `Lx/2` elements coming in on 1 array from the host CPU, and the other `Lx/2` elements on another array.  

Thus, we'd want to do some pointer arithmetic to load half of the device GPU array with data, and the other half (starting from element `Lx/2`, in 0-based counting, counting from 0) later.  

We should also do this in a "civilized manner", utilizing best practices from C++11 to make accessing a raw pointer safe.  

So say we've allocated host arrays (I'll use `std::vector` and `std::shared_ptr` from C++11 on the CPU to show how, novelly, how it can interact nicely with CUDA C/C++11 in each cases), each of size `Lx/n=Lx/2`:  

```   
// Allocate host arrays
std::vector<float> f_vec(Lx/2,1.f);
std::shared_ptr<float> sp(new float[Lx/2],std::default_delete<float[]>());   
```       

Then allocate the device GPU array, 1 big array of size `Lx`:  

```  
// Allocate problem device arrays
auto deleter=[&](float* ptr){ cudaFree(ptr); };
std::shared_ptr<float> d_sh_in(new float[Lx], deleter);
cudaMalloc((void **) &d_sh_in, Lx * sizeof(float));
```  

Then, here's how to do `cudaMemcpy` with (smart) pointer arithmetic:  

```  
cudaMemcpy(d_sh_in.get(), f_vec.data(), Lx/2*sizeof(float),cudaMemcpyHostToDevice);
cudaMemcpy(d_sh_in.get()+Lx/2, sp.get(), Lx/2*sizeof(float),cudaMemcpyHostToDevice);
```  

We can also do this with `std::unique_ptr`:  

```  
auto deleter=[&](float* ptr){ cudaFree(ptr); };

// device pointers
std::unique_ptr<float[], decltype(deleter)> d_u_in(new float[Lx], deleter);
cudaMalloc((void **) &d_u_in, Lx * sizeof(float));

cudaMemcpy(d_u_in.get(), sp.get(), Lx/2*sizeof(float),cudaMemcpyHostToDevice);
cudaMemcpy(d_u_in.get()+Lx/2, f_vec.data(), Lx/2*sizeof(float),cudaMemcpyHostToDevice);
```  

The code is available [here](https://github.com/ernestyalumni/CompPhys/blob/mobile/moreCUDA/CUDACpp11/smart_ptrs_arith.cu).



## `CUB` and CUDA Unified Memory, and then with C++11 smart pointers; CUB allows for parallel `reduce` and `scan` for a single GPU  

To use parallel `reduce` and `scan` algorithms (they are, briefly, doing summation or the product of numbers and doing a running summation, like a check book, respectively) for a *single* GPU, using `CUB` is the only way (for a library being actively updated to be optimized for the latest CUDA release).  `nccl` cannot be used to do `reduce` and `scan` for a single GPU (cf. [my stackoverflow question](https://stackoverflow.com/questions/46028541/nccl-can-we-sum-up-all-the-values-of-an-array-on-1-device-gpu-to-obtain-the-su)) 

### `CUB` with CUDA Unified Memory  

An example of using CUDA Unified Memory (with global scope) with CUB for parallel reduce on a single GPU is here, [`device_reduce_unified.cu`](https://github.com/ernestyalumni/CompPhys/blob/mobile/moreCUDA/CUDACpp11/device_reduce_unified.cu)  

```  
// Allocate arrays
__device__ __managed__ float f[Lx];
__device__ __managed__ float g; 

...

cub::DeviceReduce::Sum( d_temp_storage, temp_storage_bytes, f, &g, Lx );  
...
```

The result or output of this is this:  
```  
 temp_storage_bytes : 1
 n : 1499
 Taken to the 2th power 
 summation : 1.12388e+09
```  

Using CUDA Unified Memory with CUB (or even using CUB in general) is nontrivial because we need 2 "variables" (an array and then a single variable that'll also act as a pointer to a single value), we need to request and allocate temporary storage to find out the "size of the problem" and do `cub::DeviceReduce::Sum` twice.  

### `CUB` with C++11 smart pointers  

C++11 smart pointers makes working with `CUB` easier (or at least more organized) because:
* Use C++11 smart pointers to build in a deleter and so we don't forget to free up memory at the end of the code  
* make pointing to the raw pointer safe with `.get()`  

Look at [`device_reduce_smart.cu`](https://github.com/ernestyalumni/CompPhys/blob/mobile/moreCUDA/CUDACpp11/device_reduce_smart.cu):  
```  
// Allocate problem device arrays
auto deleter=[&](float* ptr){ cudaFree(ptr); };
std::shared_ptr<float> d_in(new float[Lx], deleter);
cudaMalloc((void **) &d_in, Lx * sizeof(float));

// Initialize device input
cudaMemcpy(d_in.get(), f_vec.data(), Lx*sizeof(float),cudaMemcpyHostToDevice);

// Allocate device output array
std::shared_ptr<float> d_out(new float(0.f), deleter);
cudaMalloc((void **) &d_out, 1 * sizeof(float));

// Request and allocate temporary storage
std::shared_ptr<void> d_temp_storage(nullptr, deleter);
	
size_t 		temp_storage_bytes = 0;

cub::DeviceReduce::Sum( d_temp_storage.get(), temp_storage_bytes, d_in.get(),d_out.get(),Lx);

cudaMalloc((void **) &d_temp_storage, temp_storage_bytes);
	
// Run
cub::DeviceReduce::Sum(d_temp_storage.get(),temp_storage_bytes,d_in.get(),d_out.get(),Lx);

```
Notice how we can use `std::shared_ptr` with CUB and not `std::unique_ptr` with CUB.  I've found (with extensive experimentation) that **it's because `CUB` needs to "share" the pointer when it's allocating the size of the problem and memory to work on given that size.**  

With `std::shared_ptr`, we can use `.get()` to get the raw pointer safely, it makes the creation and allocation of device arrays for CUB much more clearer (organized), and one can also use this with CUDA Unified Memory (I'll have to try this)  
## `nccl` and C++11 smart pointers, and as a bonus, C++11 smart pointers for CUDA *streams*.    

I have also wrapped `nccl` (briefly, it is for parallel `reduce` and `scan` algorithms, but for a multi-GPU setup) into C++11 smart pointers, for automatic cleaning up and safe pointing to the raw pointer.  
Looking at [`Ex01_singleprocess_b.cu`](https://github.com/ernestyalumni/CompPhys/blob/mobile/moreCUDA/CUDACpp11/Ex01_singleprocess_b.cu):  

```  
// managing a device
auto comm_deleter=[&](ncclComm_t* comm){ ncclCommDestroy( *comm ); };
	std::unique_ptr<ncclComm_t, decltype(comm_deleter)> comm(new ncclComm_t, comm_deleter);	

// device pointers
auto deleter=[&](float* ptr){ cudaFree(ptr); };
std::unique_ptr<float[], decltype(deleter)> d_in(new float[size], deleter);
cudaMalloc((void **) &d_in, size * sizeof(float));

std::unique_ptr<float[], decltype(deleter)> d_out(new float[size], deleter);
cudaMalloc((void **) &d_out, size * sizeof(float));

// CUDA stream smart pointer stream
auto stream_deleter=[&](cudaStream_t* stream){ cudaStreamDestroy( *stream ); };
std::unique_ptr<cudaStream_t, decltype(stream_deleter)> stream(new cudaStream_t, stream_deleter);
	cudaStreamCreate(stream.get());

...  

cudaDeviceSynchronize();

//initializing NCCL
ncclCommInitAll(comm.get(), nDev, devs);

ncclAllReduce( d_in.get(), d_out.get(), size, ncclFloat, ncclSum, *comm.get(), *stream.get() );

```  

I want to emphasize that using `std::unique_ptr` makes the freeing up of device GPU memory automatic and safe, accessing the raw pointer safe, with `.get()`.  

Then also, with (concurrent) **streams**, we can wrap those up with a C++11 smart pointer, `std::unique_ptr`, automate the freeing up of the device stream (`cudaStreamDestroy`), and make pointing to the raw pointer safe with `.get()`.  

## CUBLAS and CUDA Unified Memory Management  

One can use CUDA Unified Memory with CUBLAS.  As an example, for an array with global scope on the device GPU's unified memory, and for doing matrix multiplication `y = a1*a*x + bet*y`, where `a` is a `m x n` matrix, `x` is a `n`-vector, `y` is a `m`-vector, and `a1,bet` are scalars, then 1 can do this:  

```     
__device__ __managed__ float a[m*n];  	// a - m x n matrix on the managed device
__device__ __managed__ float x[n];  	// x - n-vector on the managed device
__device__ __managed__ float y[m];  	// y - m-vector on the managed device   

int main(void) {
	cudaError_t cudaStat;			// cudaMalloc status
	cublasStatus_t stat;  		// CUBLAS functions status
	cublasHandle_t handle; 				// CUBLAS context

	cublasCreate(&handle);
    ...  


	stat=cublasSgemv(handle,CUBLAS_OP_N,m,n,&a1,a,m,x,1,&bet,y,1);

	cudaDeviceSynchronize(); 

```  
Note the use of `cudaDeviceSynchronize()` that is necessitated if you then need to use the array on the host CPU.  

Code for this is found [here](https://github.com/ernestyalumni/CompPhys/blob/mobile/moreCUDA/CUDACpp11/014sgemv_unified.cu).  

## Short Glossary of APIs (i.e. API documentation)  

### `cudaMallocHost`  
```  
__host__ cudaError_t cudaMallocHost(void** ptr, size_t size)  
```  

Allocates page-locked memory on the host.  

**Parameters**  
`ptr` - Pointer to allocated host memory  
`size` - Requested allocation size in bytes  

**(brief) Description**

Allocates `size` bytes of host memory that is page-locked and accessible to the device.  The drive tracks the virtual memory ranges allocated with this function and automatically accelerates calls to functions such as `cudaMemcpy*()`.  Since the memory can be accessed directly by the device, it can be read or written with much higher bandwidth than pageable memory obtained with functions such as `malloc()`.  



