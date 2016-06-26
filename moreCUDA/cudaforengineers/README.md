# CUDA for Engineers

This subdirectory in `CompPhys`, `CompPhys/moreCUDA/cudaforengineers` is a "mirror" of the [cudaforengineers](https://github.com/myurtoglu/cudaforengineers) repository by, presumably, the authors of *CUDA for Engineers*, Duane Storti and Mete Yurtoglu, but with my modifications, and stripped of Visual Studio "boilerplate." 

| file or directory name |  Chapter in Storti and Yurtoglu | (Sub)Section in Storti and Yurtoglu | Description/My notes |
| ---------------------- | :-----------------------------: | :--------------------: | :--------------------------: |
| `./dd_1d_global/*`     | Chapter 5. Stencils and Shared Memory | Computing Derivatives on a 1D Grid | 1-dim. double derivatives (dd) via finite difference method with global memory |
| `./dd_1d_shared/*`     | Chapter 5. Stencils and Shared Memory | Computing Derivatives on a 1D Grid | 1-dim. double derivatives (dd) via finite difference method with shared memory |
| `./dist_3d/dist_3d.cu` | Chapter 7. Interacting with 3D Data | Launching 3D Computational Grids: `dist_3d` | This seems like a Euclidean norm implementation in 3-dimensions illustrating the use of a 3-dim. grid with 3-dim. blocks: see [my notes](https://github.com/ernestyalumni/CompPhys/blob/master/LaTeXandpdfs/CompPhys.pdf), the section on *global thread Indexing: 1-dim., 2-dim., 3-dim.* for a full generalization |
| `./dist_2d.cu`            | | | Same as `./dist_3d/dist_3d.cu`, but it's a 2-dim. Euclidean norm |


## Further notes

- For [`dd_1d_shared`](https://github.com/ernestyalumni/CompPhys/tree/master/moreCUDA/cudaforengineers/dd_1d_shared)
	* It's an instructive example on *shared memory* and *stenciling*
	* `extern` - [How do I call a C function from C++? ](https://isocpp.org/wiki/faq/mixing-c-and-cpp)
- For [`dist_3d`](https://github.com/ernestyalumni/CompPhys/tree/master/moreCUDA/cudaforengineers/dist_3d)     
[Difference between malloc and calloc?](http://stackoverflow.com/questions/1538420/difference-between-malloc-and-calloc) - `calloc()` zero-initializes the buffer, while `malloc()` leaves the memory uninitialized.

Zeroing out the memory may take a little time, so you probably want to use `malloc()` if that performance is an issue. If initializing the memory is more important, use `calloc()`. For example, `calloc()` might save you a call to `memset()`.

[Efficiency of CUDA vector types (float2, float3, float4)](http://stackoverflow.com/questions/26676806/efficiency-of-cuda-vector-types-float2-float3-float4)
