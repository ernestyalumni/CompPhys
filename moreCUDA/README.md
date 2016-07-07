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

In this `README.md`:
- Pitched Pointer, 2d array, 3d array on the device
- Finite-Difference, shared memory, tiling
- C++ Classes on the device, GPU

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

### Shared Memory

[Using Shared Memory in CUDA C/C++](https://devblogs.nvidia.com/parallelforall/using-shared-memory-cuda-cc/) by [Mark Harris](https://devblogs.nvidia.com/parallelforall/author/mharris/)

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
	* ```  
/usr/lib/gcc/x86_64-redhat-linux/5.3.1/../../../../lib64/crt1.o: In function `_start':  
(.text+0x20): undefined reference to `main'  
collect2: error: ld returned 1 exit status  
```  
- `-I.` is the short cut, short name, for the flag `--include-path` *`path`*, e.g. `-I.` is `--include-path ./` i.e. the current (working) directory: cf. [3.2.2. File and Path Specfications](http://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#axzz4DIjT7Rkc).   
- `-x cu` - "This option tells `nvcc` to treat input files as `.cu` files containing both CPU and GPU code.  By default, `nvcc` treats `.cpp` files as CPU-only code.  This option is required to have `nvcc` generate device code here, but it's also a hand way to avoid renaming source files in larger projects."  (cf. Scudiero and Murphy).  In practice, I've found that if I originally kept the `.cu` suffixes for the files, then it compiles without this flag, but this is good to know.  

-side note to keep in mind if using `#include <cuda_runtime.h>` - "if you `#include <cuda_runtime.h>` in a `.cpp` file and compile it with a compiler other than `nvcc`, `__device__` and `__host__` will be defined to nothing to enable portability of this code to other compilers!"







