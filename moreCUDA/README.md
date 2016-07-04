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

I **highly recommend** this link for reading to do C++ classes on the device *right*: [Separate Compilation and Linking of CUDA C++ Device Code](https://devblogs.nvidia.com/parallelforall/separate-compilation-linking-cuda-device-code/)

It may not be immediately evident, but from reading the article, it becomes clear that you can use 
* *both* `__host__` and `__device__` prefixes (or decorations) so class can be used on both the host **and** the device
* `__host__` *only*, by itself, so class can be used on *only* the host 
* (the most useful option, in my opinion and practice, please correct me if I'm wrong) `__device__` **only**, *by itself*, so class can be used *only* on the device - and if you want your C++ class to run on the device GPU, from a `__global__` function, this is the way to go.  This *includes* instantiating (i.e. creating) objects, arrays, stuff, etc. for your class to contain.  


This article also resolves the compilation issue(s) from others who ran into similar problems:
* [This guy had this error message: `error: calling a host function from a __device__/__global__ 
function is not allowed`](https://github.com/lvaccaro/truecrack/issues/3)
* [Work around here was to rename `.cpp` files to `.cu` - **no need to do that anymore**](https://groups.google.com/forum/#!topic/thrust-users/m9TFVWaBxkw).  Again, see the very lucid explanation in [Separate Compilation and Linking of CUDA C++ Device Code](https://devblogs.nvidia.com/parallelforall/separate-compilation-linking-cuda-device-code/) - all you have to do now is to add the `-x cu` flag, explained in that article, e.g. `nvcc -x cu myfilenamehere.cpp`


