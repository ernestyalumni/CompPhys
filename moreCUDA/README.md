# moreCUDA

`moreCUDA` in the github repository `CompPhys` contains more CUDA C/C++ examples.  

## Table of Contents

- [`cudaforengineers`](https://github.com/ernestyalumni/CompPhys/tree/master/moreCUDA/cudaforengineers) - a "mirror" of the [cudaforengineers](https://github.com/myurtoglu/cudaforengineers) repository by, presumably, the authors of *CUDA for Engineers*, Duane Storti and Mete Yurtoglu, but with my modifications, and stripped of Visual Studio "boilerplate." 
- `matmultShare.cu`  - example of using `__shared__` memory in CUDA C on GPU device
  * cf. [Matrix Multiplication with CUDA | A basic introduction
to the CUDA programming model, Robert Hochberg](http://www.shodor.org/media/content/petascale/materials/UPModules/matrixMultiplication/moduleDocument.pdf)
  * 20160620
  * more links I found
    - [5KK73 GPU assignment website 2014/2015](http://www.es.ele.tue.nl/~mwijtvliet/5KK73/?page=mmcuda)
    - [Tiled Matrix Multiplication Kernel](http://www.umiacs.umd.edu/~ramani/cmsc828e_gpusci/Lecture5.pdf)
- **Hillis/Steele (inclusive)** and **Blelloch (i.e. Prefix, exclusive) scan(s)** in subdirectory `scan`


In this `README.md`:
- `scan`, containing **Hillis/Steele (inclusive)** and **Blelloch (i.e. Prefix, exclusive) scan(s)**
- `cudaMemcpy`
- Pitched Pointer, 2d array, 3d array on the device
- `cudaMallocArray` and associated examples (in NVIDIA CUDA 8.0 Samples)
- Constant Memory, `__constant__`
- Finite-Difference, shared memory, tiling
- C++ Classes on the device, GPU
- Compiling errors when using `__constant__` memory
- Dirty CUDA C/C++ Troubleshooting

| codename        | Key code, code function, demonstrations | Description             |
| --------------- | :-------------------------------------: | :---------------------- |
| `dev3darray.cu` | `cudaMalloc3DArray`                     |                         |
| `learrays.cu`   | `__constant__`, `cudaMemcpy`, `cudaMalloc` | arrays of `float3`, on host, on device |
| `./scan/`       | scan, scans, Hillis/Steele (inclusive) scan, Blelloch (exclusive) scan, Prefix scan | Hillis/Steele (inclusive) and Blelloch (i.e. Prefix, exclusive) scan(s) |     


| Samples (NVIDIA CUDA 8.0 Samples) associated with CUDA Runtime API list   |
| ------- |
| cf. [Table 5. CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-samples/index.html#runtime-cudaapi) |


| CUDA Runtime API  | Samples | Directory (in Toolkit 8.0) | folder (in Toolkit 8.0) |
| :---------------: | :------ | -------------------------: | ----------------------  |
| `cudaMallocArray` | Pitch Linear Texture, Simple Surface Write, Simple Texture | `0_Simple` | `simplePitchLinearTexture`, `simpleSurfaceWrite`, `simpleTexture` | 

### Hillis/Steele (inclusive) scan, Blelloch (prefix; exclusive) scan(s)

In the subdirectory [`scan` in Lesson Code Snippets 3](https://github.com/ernestyalumni/cs344/tree/master/Lesson%20Code%20Snippets/Lesson%203%20Code%20Snippets/scan) is an implementation in **CUDA C++11 and C++11**, with global memory, of the *Hillis/Steele* (inclusive) scan, *Blelloch* (prefix; exclusive) scan(s), each in both parallel and serial implementation.  

As you can see, for large float arrays, running parallel implementations in CUDA C++11, where I used the GeForce GTX 980 Ti **smokes** being run serially on the CPU (I use for a CPU the *Intel® Xeon(R) CPU E5-1650 v3 @ 3.50GHz × 12*.

![scansmain](https://ernestyalumni.files.wordpress.com/2016/11/scansmainscreenshot20from202016-11-042005-12-36.png)

Note that I was learning about the Hillis/Steele and Blelloch (i.e. Prefix) scan(s) methods in conjunction with Udacity's cs344, <a href="https://classroom.udacity.com/courses/cs344/lessons/86719951/concepts/903740660923#">Lesson 3 - Fundamental GPU Algorithms (Reduce, Scan, Histogram), i.e. Unit 3.</a>.  I have a writeup of the notes I took related to these scans, formulating them mathematically, on my big <a href="https://github.com/ernestyalumni/CompPhys/blob/master/LaTeXandpdfs/CompPhys.pdf">CompPhys.pdf, Computational Physics notes</a>.    




## `cudaMemcpy`

*Note*: for some reason, you *cannot* do this (see `learrays.cu`):

```
float3* divresult
float3* dev_divresult;

cudaMalloc((void**)&dev_divresult, sizeof(float3));

// do stuff on dev_divresult from a `__global__` function

cudaMemcpy( divresult, dev_divresult, sizeof(float3), cudaMemcpyDeviceToHost) ;

```  

I obtain Segmentation Faults when trying to read out the result.  

I **can** do this:

```
**float3 divresult**
float3* dev_divresult;

cudaMalloc((void**)&dev_divresult, sizeof(float3));

// do stuff on dev_divresult from a `__global__` function

**cudaMemcpy( &divresult, dev_divresult, sizeof(float3), cudaMemcpyDeviceToHost) ;**

```  

cf. [CUDA invalid argument when trying to copy struct to device's memory (cudaMemcpy)q](http://stackoverflow.com/questions/24460507/cuda-invalid-argument-when-trying-to-copy-struct-to-devices-memory-cudamemcpy)



## Pitched Pointer, 2d array, 3d array on the device

`pitcharray2d.cu` is an implementation of the code in [(2012 Jan 21) *Allocating 2D Arrays in Cuda* By Steven Mark Ford](http://www.stevenmarkford.com/allocating-2d-arrays-in-cuda/).  

From [CUDA Runtime API, CUDA Toolkit Documentation](http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g32bd7a39135594788a542ae72217775c)

```  
__host__ cudaError_t cudaMallocPitch( void** devPtr, size_t* pitch, size_t width, size_t height)  
```  

`pitch` is, as a separate parameter of memory allocation, used to compute addresses within the 2D array.  The pitch returned in `*pitch` by `cudaMallocPitch()` is the width in bytes of the allocation.  

For `pitcharray2db.cu`, I appear to obtain the fastest calculation of a 1000x1000 array on the device, squaring individual elements, with grid, block dimensions, in x-direction, of 32,64 (32 blocks on the grid in the x-direction, 64 threads on a block in the x-direction).  

For `pitcharray2dc.cu`, I launch on 1 block, 1 thread, because when I do `printf`, `printf` executes once; whereas if I launch on 2 blocks, 2 threads, `printf` executes a total of 4 times.  Where does this device (2-dimensional) array live, does it exist on the global memory only once?  

```  
__host__ cudaError_t cudaMalloc3D( cudaPitchedPtr* pitchedDevPtr, cudaExtent extent )  
```

`pitchedDevPtr` is a pointer to allocated pitched device memory.  It appears that from `pitchedDevPtr`, type cudaPitchedPtr has members (I think it might be a struct, or "class-like") `.ptr` and `.pitch`.  

`extent` is a `cudaExtent` struct (it's a very easy, basic struct) that has the dimensions.  

Then this [part of the CUDA Toolkit v7.5 documentation, Programming Guide](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#axzz4DfygExhf) had a code snippet implementing this.  I've put this in `pitcharray3d.cu`.  

However, in the documentation, there's a `(char *)` pointer to a char, but `cudaPitchedPtr` instance, called `pitchedDevPtr` in this case, has, as a data field, `.ptr`, but it's a **`void *`**, i.e. pointer to a void (!!!).  

cf. [cudaPitchedPtr Struct Reference](http://horacio9573.no-ip.org/cuda/structcudaPitchedPtr.html)
Data Fields:
```  
size_t pitch
void* ptr
size_t xsize
size_t ysize
```
What I did was to force the type from `(void *)` to `(char *)`.  

For a 64x64x64 grid, I get the fastest times (68.0802 ms, about) for grid, block dims. (32,4)  

```  
__host__ cudaError_t cudaMalloc3DArray( cudaArray_t* array, const cudaChannelFormatDesc* desc, cudaExtent extent, unsigned int flags=0)
```

`cudaChannelFormatDesc` appears to be a struct.  It seems that we could think of $(x,y,z,w)$ as a 4-vector.  However, memory has to be accounted for in *bits*, not bytes.  

`cudaChannelFormatDesc` is defined as (cf. CUDA Runtime API, `CUDA_Runtime_API.pdf`)
```  
struct cudaChannelFormatDesc {
	int x, y, z, w;
	enum cudaChannelFormatKind
		f;
};
```  
where `cudaChannelFormatKind` is one of `cudaChannelFormatKind` is one of `cudaChannelFormatKindSigned`, `cudaChannelFormatKindUnsigned`, or `cudaChannelKindFloat`.  

I like to think that a $C^{\infty}(\mathbb{R}^3)$ function on $\mathbb{R}^3$ can be represented as follows (in this context):

```
cudaChannelFormatDesc somenameChannel { CHAR_BIT*sizeof(float), 0, 0, 0, cudaChannelKindFloat }
```
with `CHAR_BIT` being the size of 1 byte in bits and `sizeof(float)` being the size of a `float` in bytes.  

#### On cudaArray , straight from the horses mouth:

[3.2.11.3. CUDA Arrays, CUDA Toolkit 7.5 Documentation](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-arrays) "CUDA arrays are opaque memory layouts optimized for texture fetching. They are one dimensional, two dimensional, or three-dimensional and composed of elements, each of which has 1, 2 or 4 components that may be signed or unsigned 8-, 16-, or 32-bit integers, 16-bit floats, or 32-bit floats. **CUDA arrays are only accessible by kernels through texture fetching as described in Texture Memory or surface reading and writing as described in Surface Memory**."

### Texture memory

[Textures from Moshovos, http://www.eecg.toronto.edu/~moshovos/CUDA08/slides/008 - Textures.ppt (1.742 Mb)](http://www.eecg.toronto.edu/~moshovos/CUDA08/doku.php?id=lecture_slides)

**If** you cannot write to texture memory, but only be able to write to surface memory, then I will implement in surface memory.  

### Surface Memory

cf. [3.2.11.2.2. Surface Reference API](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#surface-reference-api)

surface reference is declared at file scope (at the top?) as a variable of type surface:
```
surface<void, Type> surfRef;
```
where `Type` species type of surface reference and is equal to 
- `cudaSurfaceType1D`
- `cudaSurfaceType2D`
- `cudaSurfaceType3D`
- `cudaSurfaceTypeCubemap`
- `cudaSurfaceType1DLayered`
- `cudaSurfaceType2DLayered`
- `cudaSurfaceTypeCubemapLayered`

`surfRef` is an arbitrary name (can be `outputSurf`, etc.)



### Shared Memory  

[Using Shared Memory in CUDA C/C++](https://devblogs.nvidia.com/parallelforall/using-shared-memory-cuda-cc/) by [Mark Harris](https://devblogs.nvidia.com/parallelforall/author/mharris/)

## `cudaMallocArray` and associated examples (in NVIDIA CUDA 8.0 Samples)

## Constant Memory, `__constant__`

Here's how to use `__constant__` memory in the header file and definition: cf. [Constant memory symbols](https://devtalk.nvidia.com/default/topic/569901/constant-memory-symbols/?offset=1) , from user [hrvthzs](https://devtalk.nvidia.com/member/1919793/) 

Somebody answered my question here:
(http://stackoverflow.com/questions/18213988/cuda-constant-memory-symbols)

So at first you need to create a header file and declare the symbol:
```  
    // constant.cuh
    extern __constant__ var_type var_name;
```  

then define only once the symbol in a .cu file:
```  
    // constant.cu
    __constant__  var_type var_name;
``` 

If you need this symbol in other `.cu` files, just include the header. 


*My experience* with `__constant__` constant memory : it's read-only, cached memory on the GPU.  I think of it as when I'm working on a physics problem, and I need to plug in a few ($<21$) parameters into my physics problem (physical constants, physical parameters, etc.), then I'll need those constants all the type, floating around.  I would use constant memory, `__constant__` for those values.

`__constant__` memory stuff (objects) have to be in global scope.  So it's at the top of the "`main.cu`" file.  Or, as I've found, they're `extern __constant__ somearray[N]` in your header file (e.g. 'thisismylibrarywithclasses.h') and then as `__constant__ somearray[N]` in your definition file ('thisismylibrarywithclasses.cu`).  I use them as **global** constants anywhere in my code, in particular, in the main.

I've found that I *cannot* get away with passing those globally defined, in constant memory, objects (such as arrays), into a `__global__` device function.  I've found that I *can* pass them in as arguments for a `__device__` device function.

However, when I think about it, if the program is such that it's running on the device GPU, I'm looking down from `__global__` memory, to `__shared__` memory and local (thread block) memory, etc., and the constant exists globally, then I don't think it makes sense to pass them in as arguments.  The `__device__ functions should just use them, i.e.

$$
`__constant__` \in `__global__`, \text{ in fact } `__constant__` \in \text{Obj}{`__global__`} 
$$

then it makes no sense to "subset", or do restriction functor on  `__constant__` objects down to your routine.  

- From your `__global__` device function, you *CANNOT* pass in the `__constant__` objects (stuff) as arguments; just use them *directly* since they're already sitting on a "global", special read-only cached memory
- From your `__device__` function, you could pass them in as arguments (but why?)




## Finite-Difference, shared memory, tiling

I asked this on stackoverflow.  

Could someone please help explain tiling, the tile method for shared memory on the device GPU, with CUDA C/C++, in general?  

There are a number of Powerpoint lecture slides in pdf that I've found after a Google search, namely [Lecture 3 Shared Memory Implementation part, pp. 21](http://www.bu.edu/pasi/files/2011/07/Lecture31.pdf) and/or [Zahran's Lecture 6: CUDA Memories](http://cs.nyu.edu/courses/spring12/CSCI-GA.3033-012/lecture6.pdf).  However, these *Powerpoint* slides (obviously) don't give a full, written explanation and the books out there are 3-4 years old (and there are huge changes in CUDA C/C++ and Compute Capability now being at 5.2, since then).    

Here's how I understand using shared memory for CUDA C so far (I know stackoverflow doesn't support LaTeX for math expressions, which I understand, given the HUGE dependencies (bulkiness?) of LaTeX; I've done a write up in LaTeX [here](https://github.com/ernestyalumni/CompPhys/blob/master/LaTeXandpdfs/CompPhys.pdf), the section called "Note on finite-difference methods on the shared memory of the device GPU, in particular, the pencil method, that attempts to improve upon the double loading of boundary “halo” cells (of the grid)").

## Setup, `__global__` memory loading

Suppose your **goal** is to compute on a 3-dimensional grid, ($\equiv targetgrid3d$) of size $N_x \times N_y \times N_z = N_xN_yN_z$  e.g. $N_x = N_y = N_z = 800$.  Flattened, this would be a $N_xN_yN_z = 512000000$ sized 1-dimensional array, say of floats, and both the CPU and GPU should have enough RAM to load this (on my workstation, it has 32 GB CPU RAM, on the NVIDIA GeForce GTX 980Ti, there's [8 GB of system memory](http://www.geforce.com/hardware/geforce-gtx-980-ti/buy-gpu).  As I'd want to calculate on values at each point of this 3-dimensional grid, targetgrid3d, I'd want to launch a kernel function (call) on a total of $N_xN_yN_z$ threads. 


However, the computation I'm doing is such that, $\forall \, $ pt on that targetgrid3d,$\equiv k$, $k\in \mathbb{Z}^+$, $k=\lbrace 0, \dots N_xN_yN_z \rbrace$, i.e. for each grid point, each computation involving single $k$ involves only a (much smaller than N_xN_yN_z) finite number of other points on targetgrid3d: for example, for 3-dimensional finite difference of the 0th order (I think that's the right number, for 0), it involves the "left", "right", "up", "down", "bottom" and "top" grid pts. adjacent to $k$: 6+1 in total.  Let's just denote $6+1=R$.  

I appreciate the rationals given of how doing this computation in `__global__` memory is costly, including the fact that we're copying $R$ values each computation, $\forall k \in \lbrace 0 \dots N_xN_yN_z-1\rbrace$.  

Let $M_i := $ number of threads per block in the $i$th direction, $i = x,y,z$.  
Let $N_i^{block} :=$ number of blocks launched on the **memory** grid (*NOT* our targetgrid3d) in the $i$th direction, $i=x,y,z$.  Then the total number of threads in the $i$ direction is $M_i*N_i^{block}$ This *can be* different (let's say smaller) than our goal, our targetgrid3d, in each dimension, i.e. $N_i > M_i*N_i^{block}$.  For example, we can have `dim3` $(M_x,M_y,M_z) = (4,4,4)$ for our (block) thread dimensions (i.e. `blockdim.x, blockdim.y, blockdim.z`), and launch blocks $(N_x^{block},N_y^{block},N_z^{block}) = (N_x/M_x, N_y/M_y, N_z/M_z) = (200, 200, 200)$ for example.  I'll point out 2 things at this point:  
- you can get the needed blocks from the formula 
\[
\text{ block needed in the ith direction } = (N_i +M_i -1)/M_i
\]
integer arithmetic works out that you get equal or more blocks needed to run $N_i$ threads in the $i$th direction that you desire.
- desired $N_i$ can be much larger than $N_i^{block}*M_i$, with $N_i^{block}$ being limited by the maximum number of blocks that can be launched (which can be found from `cudaGetDeviceProperties` or by running my small CUDA C script, [queryb.cu](https://github.com/ernestyalumni/CompPhys/blob/master/CUDA-By-Example/queryb.cu).  In particular, for my GTX 980Ti, I get this read out:

```  
    --- General Information for device 0 ---
Name: GeForce GTX 980 Ti
Compute capability: 5.2
Clock rate:  1076000
Device copy overlap:  Enabled
Kernel execution timeout :  Enabled
   --- Memory Information for device 0 ---
Total global mem:      6441730048
Total constant Mem:    65536
Max mem pitch:         2147483647
Texture Alignment:     512
   --- MP Information for device 0 ---
Multiprocessor count:  22
Shared mem per mp:     49152
Registers per mp:      65536
Threads in warp:       32
Max threads per block: 1024
Max thread dimensions: (1024, 1024, 64) 
Max grid dimensions:   (2147483647, 65535, 65535) 
```  
So for instance, $M_x*M_y*M_z$, total number of threads in a block, is limited to a maximum of 1024.  

## "Naive" shared memory method, halo cells, "radius" $r$ of halo cells on a (single) thread block 

Then for (what I call) the "naive" shared memory method, what one wants to do, for each thread block of size $M_x*M_y*M_z$ is to load only the values involved in computation into *shared memory*, `__shared__`.  For example, for the 2-dimensional problem (2-dim. targetgrid), considering a thread block of $(M_x,M_y)$, you're going to want the $(M_x+2r)*(M_y+2r)$ values to be shared on that thread block, $r = 1$ in a $0$th order finite difference case, with $r$ denoting the so-called "halo" cells ("halo" being the terminology that seems to be used in teaching GPU shared memory).  I'm supposing that this generalizes in 3-dim., the 3-dim. thread block $(M_x,M_y,M_z)$ to needing $(M_x+2r)*(M_y+2r)*(M_z+2r)$ values to go into shared memory.  

The factor of 2 comes from needing values "outside" the thread block on "both" sides.  I call this (and my idea of this, correct me if I'm wrong) dealing with *boundary conditions* for shared memory (on (a single) thread block).  

To make this concrete, a code that implements this shared memory idea in 2-dimensions, something that solves the heat equation to $0$th order precision, is [heat_2d](https://github.com/myurtoglu/cudaforengineers/tree/master/heat_2d) from Storti and Yurtoglu (it's nice that their respective book, **CUDA for Engineers**, came out last year, 2015 - I don't have the money to purchase a copy, but their github repository is useful).  As you can see in the code [kernel.cu](https://github.com/myurtoglu/cudaforengineers/blob/master/heat_2d/kernel.cu) with the "meat" of the code being in `__global__ void tempKernel`, the "boundary condition" or "corner cases" for cells on the "edge" of the thread block, are treated correctly, to "grab" values outside the thread block in question (Lines 53-64, where the `//Load "halo" cells` comment starts), and, namely, the so-called "radius" of halo cells, $r=1$ (`#define RAD 1`) is clearly defined.

## What's tiling?

So what's tiling?  From what I could glean from lecture slides, it appears to be trying to "load more data than threads" on each thread block.  Then, presumably, a `for` loop is needed to run on each thread block.  But for our problem of the targetgrid3d of size $(N_x,N_y,N_z)$, does this mean we'd want to launch threads in a block `dim3` $(M_x,M_y,M_z)$, grid blocks `dim3` $(N_x^{block}, N_y^{block}, N_z^{block})$ such that, for example, $2*M_i*N_i^{block} = N_i$?  So each thread block does 2 computations because we're loading in twice the number of data to go into shared memory? (2 can be 3, 4, etc.)

How then do we deal with "boundary condition" "halo" cells in this case of tiling, because after "flattening" to a 1-dimensional array, it's fairly clear how much "stride" we need to reach each cell on our targetgrid3d (i.e. $k = k_x + k_y N_xM_x + k_z N_xM_xN_yM_y$, with $k_x = i_x + M_x*j_x \equiv $ `threadIdx.x + blockDim.x*blockIdx.x`, see [Threads, Blocks, Grids section](https://github.com/ernestyalumni/CompPhys/blob/master/LaTeXandpdfs/CompPhys.pdf)).  How do we get the appropriate "halo" cells needed for a "stencil" in this case of tiling?  

## 3-dim. Finite Difference, the "pencil" method, scaling the pencil method, can it work? 

Harris proposes, in (Finite Difference Methods in CUDA C/C++, Part 1)[https://devblogs.nvidia.com/parallelforall/finite-difference-methods-cuda-cc-part-1/] a "pencil" method for utilizing shared memory as an improvement upon the described-above "naive" shared memory.  

I point out that there are 2 problems with this, as this method doesn't scale at all - 1. the thread block size easily exceeds the maximum thread block size as you're require the full $N_i \times s_{pencil}$, $s_{pencil} =4$ for example, for the thread size $M_i$ to be launched, and 2. the shared memory required is the halo radius (in this case for 8th order finite difference, about 4 adjacent cells, $r=4$) plus the full dimension size of targetgrid3d, $N_i$ (he denotes it as `m*`, e.g. `mx,my,mz`, in the [finite_difference.cu](https://github.com/parallel-forall/code-samples/blob/master/series/cuda-cpp/finite-difference/finite-difference.cu) code.  
 
In fact, as a concrete example, I take the [finite_difference.cu](https://github.com/parallel-forall/code-samples/blob/master/series/cuda-cpp/finite-difference/finite-difference.cu) code and you can change $N_x=N_y=N_z$ from 64 to 354, `mx=my=mz=64` to 354 in Harris' notation.  Above that, one obtains Segmentation Faults.  Clearly, for `derivative_x`, the derivative in the $x$ direction, with  

(my/spencils, mz,1) for the grid dimensions (number of blocks in each dimension) and  
(mx,spencils,1) for the block dimensions (number of threads in each (thread) block)

Then you'd have mx*spencils*1 threads to launch on each block, which is severely limited by the maximum number of threads per block. (e.g. 354*4*1 = 1416 > 1024).  

However, in the "naive" shared memory method, you can keep the thread block size < 1024, $(M_x,M_y,M_z)$, small enough, and the shared memory small enough, $(M_x+2r)*(M_y+2r)*(M_z+2r)$.  

Does this pencil method not scale?  Harris mentions to decouple the  total domain size from the tile size, in the comments. He further explains that "Threads can execute loops, so you can loop and process multiple tiles of the entire domain using a single thread block; don't try to fit the entire domain in shared memory at once."  

* How would I "partition up" the targetgrid3d, called total domain here?    
* The mapping from targetgrid3d to each threadIdx.*, to blockIdx.* is fairly clear with the "naive" shared memory method, but how does that all have to change with tiling?  
* How are thread block boundary conditions, on the "halo" cells dealt with in tiling?

For a concrete example, I don't know how to modify [finite_difference.cu](https://github.com/parallel-forall/code-samples/blob/master/series/cuda-cpp/finite-difference/finite-difference.cu) code to solve derivatives for a grid larger than (354^3).

### mapping target grid3d, the "domain" to 2-dimensional device GPU memory

From Harris' article on Finite Difference, it appears that the 3-dimensional domain is mapped to the 2-dimensional device GPU memory, device memory.  I can see that advantage of this as the max. number of blocks that can be launched in the $x$-direction, in particular, is much greater than in the $z$-direction (2147483647 > 65535).  Also, the max. number of threads in a block in the $x$,$y$-directions are much greater than in the $z$-direction (1024,1024>64). However, it's unclear to me how a cell index on the target 3d grid maps to the 2-dimensional device GPU memory, in the case of tiling pencils or not even tiling pencils.  How could I map the 3-dim. domain to the 2-dim. device, including the boundary cases of halo cells for a stencil, and with "tiling" or running `for` loops?   
 
 


## CUDA Thread Indexing Cheatsheet - includes 3d grid of 3d blocks!

- [CUDA Thread Indexing Cheatsheet](https://cs.calvin.edu/courses/cs/374/CUDA/CUDA-Thread-Indexing-Cheatsheet.pdf).  Note that I generalize this in the Threads, Blocks section of my notes `CompPhys.pdf`

## C++ Classes on the device, GPU

Requires Compute Capability >5.0.  

I ran [`queryb.cu`](https://github.com/ernestyalumni/CompPhys/blob/master/CUDA-By-Example/queryb.cu) 

I **highly recommend** this link for reading to do C++ classes on the device *right*: [Separate Compilation and Linking of CUDA C++ Device Code](https://devblogs.nvidia.com/parallelforall/separate-compilation-linking-cuda-device-code/) from Tony Scudiero and Mike Murphy. 

It may not be immediately evident, but from reading the article, it becomes clear that you can use 
* *both* `__host__` and `__device__` prefixes (or decorations) so class can be used on both the host **and** the device
* `__host__` *only*, by itself, so class can be used on *only* the host 
* (the most useful option, in my opinion and practice, please correct me if I'm wrong) `__device__` **only**, *by itself*, so class can be used *only* on the device - and if you want your C++ class to run on the device GPU, from a `__global__` function, this is the way to go.  This *includes* instantiating (i.e. creating) objects, arrays, stuff, etc. for your class to contain.  


This article also resolves the compilation issue(s) from others who ran into similar problems:
* [This guy had this error message: `error: calling a host function from a __device__/__global__ 
function is not allowed`](https://github.com/lvaccaro/truecrack/issues/3)
* [Work around here was to rename `.cpp` files to `.cu` - **no need to do that anymore**](https://groups.google.com/forum/#!topic/thrust-users/m9TFVWaBxkw).  Again, see the very lucid explanation in [Separate Compilation and Linking of CUDA C++ Device Code](https://devblogs.nvidia.com/parallelforall/separate-compilation-linking-cuda-device-code/) - all you have to do now is to add the `-x cu` flag, explained in that article, e.g. `nvcc -x cu myfilenamehere.cpp`

### `nvcc` compiler flags (compiler options) - what they mean

- `-dc` - "tells `nvcc` to generate device code for later linking" (Scudiero and Murphy);  I found that I needed `-dc` for my usual C++ classes, run on the "host" CPU, i.e. `.cpp` files with class definitions.  Otherwise, this error was obtained:  
	* 
```  
/usr/lib/gcc/x86_64-redhat-linux/5.3.1/../../../../lib64/crt1.o: In function `_start':  
(.text+0x20): undefined reference to `main'  
collect2: error: ld returned 1 exit status  
  ```  
`-dc` is short for `--device-c`, and from [3.2.1. Options for Specifying Compilation and Phase, CUDA Toolkit v7.5 NVCC documentation](http://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#nvcc-command-options), "compiles each `.c/.cc/.cpp/.cxx/.cu` input file into an object file that contains relocatable device code.  It's equivalent to `--relocatable-device-code=true --compile`


- `-I.` is the short cut, short name, for the flag `--include-path` *`path`*, e.g. `-I.` is `--include-path ./` i.e. the current (working) directory: cf. [3.2.2. File and Path Specfications](http://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#axzz4DIjT7Rkc).   
- `-x cu` - "This option tells `nvcc` to treat input files as `.cu` files containing both CPU and GPU code.  By default, `nvcc` treats `.cpp` files as CPU-only code.  This option is required to have `nvcc` generate device code here, but it's also a hand way to avoid renaming source files in larger projects."  (cf. Scudiero and Murphy).  In practice, I've found that if I originally kept the `.cu` suffixes for the files, then it compiles without this flag, but this is good to know.  

-side note to keep in mind if using `#include <cuda_runtime.h>` - "if you `#include <cuda_runtime.h>` in a `.cpp` file and compile it with a compiler other than `nvcc`, `__device__` and `__host__` will be defined to nothing to enable portability of this code to other compilers!"
- `-g` is short for `--debug` - "generate debug information for host code"
- `-G` is short for `--device-debug` - "generate debug information for device code"


## Compiling errors when using `__constant__` memory

['cicc' compilation error and debug flag](https://devtalk.nvidia.com/default/topic/527307/-39-cicc-39-compilation-error-and-debug-flag/)

I obtain a similar error when I try to "link together" or have header file dependencies on another header file and definition, when using `__constant__` memory: this appears to be a problem with the `nvcc` compiler itself and will have to fixed by NVIDIA themselves.  

> Signal 11 would indicate a memory access out of bounds, which should not happen and would point to a bug inside the compiler. 

- [njuffa](https://devtalk.nvidia.com/member/1738298/)

Adding `-G` compiler flag helps but slows down the kernel run time.  

### Dealing with the error of not being able to relocate code (problem is linking up `__device__` code) 

Useful links:

[Unable to decipher nvlink error](http://stackoverflow.com/questions/21173232/unable-to-decipher-nvlink-error)



Use `c++filt` to demangle the names. For instance:
```
    $ c++filt _ZN5JARSS15KeplerianImpactC1ERKdS2_S2_S2_S2_S2_ JARSS::KeplerianImpact::KeplerianImpact(double const&, double const&, double const&, double const&, double const&, double const&)
```  
cf. [Roger Dahl](http://stackoverflow.com/users/442006/roger-dahl)

http://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#using-separate-compilation-in-cuda

http://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#examples

http://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#nvcc-command-options





## Dirty CUDA C/C++ Troubleshooting

I found that my CUDA C/C++ program was outputting nonsense even after using `__syncthreads()` correctly (e.g. github repository `Propulsion/CUDACFD/convect1dfinitediff/`).  What I did to troubleshoot this was to change the number of threads on a block to 1, then do make, and run it again, and it works.  Then I changed the number of threads for the number of threads on a block, $M_x,M_y,M_z$, to my desired amount.  


