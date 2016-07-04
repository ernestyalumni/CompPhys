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
- C++ Classes on the device, GPU

### Shared Memory

[Using Shared Memory in CUDA C/C++](https://devblogs.nvidia.com/parallelforall/using-shared-memory-cuda-cc/) by [Mark Harris](https://devblogs.nvidia.com/parallelforall/author/mharris/)

## Finite-Difference



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







