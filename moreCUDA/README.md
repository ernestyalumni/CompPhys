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
- Texture memory explanation and some examples in [`./samples02`](https://github.com/ernestyalumni/CompPhys/tree/master/moreCUDA/samples02)

In this `README.md`:
- `scan`, containing **Hillis/Steele (inclusive)** and **Blelloch (i.e. Prefix, exclusive) scan(s)**
- `cudaMemcpy`
- Pitched Pointer, 2d array, 3d array on the device
- `cudaMallocArray` and associated examples (in NVIDIA CUDA 8.0 Samples)
- Texture memory
- Surface memory
- Constant Memory, `__constant__`
- Finite-Difference, shared memory, tiling
- C++ Classes on the device, GPU
- Compiling errors when using `__constant__` memory
- Dirty CUDA C/C++ Troubleshooting
- `thrust`, and *useful links* for *`thrust`*
- *CUB*
- Examples of using *cuBLAS*, *cuSolver*

| codename        | Key code, code function, demonstrations | Description             |
| --------------- | :-------------------------------------: | :---------------------- |
| `dev3darray.cu` | `cudaMalloc3DArray`                     |                         |
| `learrays.cu`   | `__constant__`, `cudaMemcpy`, `cudaMalloc` | arrays of `float3`, on host, on device |
| `./scan/`       | scan, scans, Hillis/Steele (inclusive) scan, Blelloch (exclusive) scan, Prefix scan | Hillis/Steele (inclusive) and Blelloch (i.e. Prefix, exclusive) scan(s) |
| `./samples02/tex1dlinearmem.cu` | `texture<,,>`, `tex1Dfetch`,`cudaBindTexture` | texture memory of 1-dim. linear array, cf. [CUDA Advanced Memory Usage and Optimization, Yukai Hung, National Taiwan Univ.](http://www.math.ntu.edu.tw/~wwang/mtxcomp2010/download/cuda_04_ykhung.pdf) |    
| `./samples02/tex1dlinearmemb.cu` | `texture<,,>`, `tex1Dfetch`,`cudaBindTexture` | texture memory of 1-dim. linear array, same as `tex1dlinearmem.cu`, but with print out of results (sanity checks) |    
| `./samples02/tex2dcuArray.cu` | `texture<,,>`, `tex2D`, `cudaArray`, `cudaChannelFormatDesc`, `cudaCreateChannelDesc`, `cudaMallocArray`, `.filterMode`, `.addressMode` | texture float memory over 2-dimensional cuda Array |
| `./samples02/tex3dcuArray.cu` | `tex3D` | texture float memory over 3-dimensional cuda Array |
| `./samples02/simpletransform.cu` | `cudaTextureObject_t`, `cudaCreateChannelDesc` | code sample, in *VERBATIM*, (attempting) to apply some simple transformation kernel to a texture cf. [3.2.11.1.1. Texture Object API of CUDA Toolkit 8 Documentation](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#abstract) |
| `./thruststuff/reduce_eg.cu` | `thrust::reduce`, `thrust::sequence`, `thrust::device_vector`, `thrust::host_vector` | Uses `reduce` to sum up all elements of a vector "directly" on the device GPU; uses `thrust::sequence` to make up some non-trivial initial values; I needed to test out `thrust`'s *reduce* algorithm | 
| `main.cu`	   | C++ class templates, `template< >`  | `./thruststuff/ranges`, cf. [Separate C++ Template Headers (`*.h`) and Implementation files (`*.cpp`)](http://blog.ethanlim.net/2014/07/separate-c-template-headers-h-and.html), other than examples of ranges, this is an example of separating C++ class templates to the header file |
| `cuRAND_eg.cu` | `cuRAND`, `CUB`, `std::vector`, `std::vector::data` | example use of **`cuRAND`** |
| [`./CUBLAS/](https://github.com/ernestyalumni/CompPhys/tree/master/moreCUDA/CUBLAS) | `cuBLAS`, CUDA Unified Memory Management, `__managed__` | Examples of using `cuBLAS`, for linear algebra, including examples for using `cuBLAS` with CUDA Unified Memory Management | 
||||
| [`./CUSOLVER/SVD_vectors.cu`](https://github.com/ernestyalumni/CompPhys/blob/master/moreCUDA/CUSOLVER/SVD_vectors.cu) | Singular Value Decomposition, SVD, `cuSOLVER`  | simple example in C of Singular Value Decomposition, but with singular vectors, cf. [CUDA Toolkit Doc, E.1 SVD with singular vectors](http://docs.nvidia.com/cuda/cusolver/index.html#svd-example1)  |


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

## Texture memory

One useful explanation: [Textures from Moshovos, http://www.eecg.toronto.edu/~moshovos/CUDA08/slides/008 - Textures.ppt (1.742 Mb)](http://www.eecg.toronto.edu/~moshovos/CUDA08/doku.php?id=lecture_slides)

[`./samples02/tex1dlinearmemb.cu`](https://github.com/ernestyalumni/CompPhys/blob/master/moreCUDA/samples02/tex1dlinearmemb.cu) is an instructive, pedagogical, and simple example of texture memory, over linear memory.  Let's take a look at what stuff means for [*texture reference*](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-reference-api).  

From the [Texture Reference API documentation, 3.2.11.1.2.](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-reference-api): texture reference is declared at file scope as variable of type texture:
```
texture<DataType, Type, ReadMode> texRef;
```
where:

* `DataType` specifies type of texel;  
* `Type` species type of texture reference and is equal to `cudaTextureType1D`, `cudaTextureType2D`, or `cudaTextureType3D`, for a one-dimensional, two-dimensional, or three-dimensional texture, respectively, or `cudaTextureType1DLayered` or `cudaTextureType2DLayered` for a one-dimensional or two-dimensional layered texture respectively; `Type` is an optional argument which defaults to `cudaTextureType1D`;  
* `ReadMode` specifies the read mode; it is an optional argument which defaults to `cudaReadModeElementType`.

A texture reference can only be declared as a static global variable and cannot be passed as an argument to a function.

e.g. from [`./samples02/tex1dlinearmemb.cu`](https://github.com/ernestyalumni/CompPhys/blob/master/moreCUDA/samples02/tex1dlinearmemb.cu), and [CUDA Pro Tip:Kepler Texture Objects Improve Performance and Flexibility | Parallel Forall](https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-kepler-texture-objects-improve-performance-and-flexibility/)
```
texture<float,1,cudaReadModeElementType> texreference
```

- **`cudaBindTexture`**, cf. [4.28. C++ API Routines](http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__HIGHLEVEL.html#group__CUDART__HIGHLEVEL_1gfaa25560127f9feb99cb5dd6bc4ce2dc)
```
template < class T, int dim, enum cudaTextureReadMode readMode >
__host__ ​cudaError_t cudaBindTexture ( size_t* offset,
	 	     		       const texture < T, dim, readMode > & tex,
				       const void* devPtr, size_t size = UINT_MAX ) [inline]
```
*Parameters*  
`offset`  
    - Offset in bytes  
`tex`  
    - Texture to bind   
`devPtr`  
    - Memory area on device   
`size`  
    - Size of the memory area pointed to by devPtr  

*Description*
Binds `size` bytes of the memory area pointed to by `devPtr` to texture reference `tex`. The channel descriptor is inherited from the texture reference type. The `offset` parameter is an optional byte offset as with the low-level `cudaBindTexture( size_t*, const struct textureReference*, const void*, const struct cudaChannelFormatDesc*, size_t)` function. Any memory previously bound to `tex` is unbound.

e.g.
```
float* diarray;
const int ARRAY_SIZE=3200;

cudaBindTexture(0, texreference, diarray, sizeof(float)*ARRAY_SIZE); // or

```

- *`tex1Dfetch()`* vs. **`tex1D()`, `tex2D()`,`tex3D()`**

cf. [9.2.4.1. Additional Texture Capabilities](http://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#additional-texture-capabilities)
"If textures are fetched using `tex1D()`, `tex2D()`, or `tex3D()` rather than `tex1Dfetch()`, hardware provides other capabilities that might be useful for some applications", some of those applications listed in Table 4, filtering (fast, low-precision interpolation between texels), normalized texture coordinates (resolution-independent coding), addressing modes (automatic handling of boundary cases).  

From [B.8.1. Texture Object API](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-object-api-appendix),
     * **`tex1Dfetch()`**
```
template<class T>
T tex1Dfetch(cudaTextureObject_t texObj, int x);
```
fetches from the region of linear memory specified by the one-dimensional texture object `texObj` using integer texture coordinate x. `tex1Dfetch()` only works with non-normalized coordinates, so only the border and clamp addressing modes are supported. It does not perform any texture filtering. For integer types, it may optionally promote the integer to single-precision floating point.
	* **`tex1D()`**
```
template<class T>
T tex1D(cudaTextureObject_t texObj, float x);
```
fetches from the CUDA array specified by the one-dimensional texture object `texObj` using texture coordinate x.
- **`cudaTextureDesc`**  
cf. [Programming Interface, CUDA Toolkit Documentation](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#axzz4PoIffz3h)

The other attributes of a texture reference are mutable and can be changed at runtime through the host runtime. As explained in the reference manual, the runtime API has a *low-level* C-style interface and a *high-level* C++-style interface. The `texture` type is defined in the high-level API as a structure publicly derived from the `textureReference` type defined in the low-level API, `textureReference`:

```    
struct textureReference {
    int                          normalized;
    enum cudaTextureFilterMode   filterMode;
    enum cudaTextureAddressMode  addressMode[3];
    struct cudaChannelFormatDesc channelDesc;
    int                          sRGB;
    unsigned int                 maxAnisotropy;
    enum cudaTextureFilterMode   mipmapFilterMode;
    float                        mipmapLevelBias;
    float                        minMipmapLevelClamp;
    float                        maxMipmapLevelClamp;
}    
```  

* `normalized` specifies whether texture coordinates are normalized or not;   
* `filterMode` specifies the filtering mode;   
* `addressMode` specifies the addressing mode;  
* `channelDesc` describes the format of the texel; it must match the DataType argument of the texture reference declaration; `channelDesc` is of the following type:

```
    struct cudaChannelFormatDesc {
      int x, y, z, w;
      enum cudaChannelFormatKind f;
    };
```

where `x`, `y`, `z`, and `w` are equal to the number of bits of each component of the returned value and f is:  
* `cudaChannelFormatKindSigned` if these components are of signed integer type,  
* `cudaChannelFormatKindUnsigned` if they are of unsigned integer type,  
* `cudaChannelFormatKindFloat` if they are of floating point type.  

  
The `cudaTextureDesc struct` is defined as  
```
‎        struct cudaTextureDesc {
                  enum cudaTextureAddressMode 
                  addressMode[3];
                  enum cudaTextureFilterMode  
                  filterMode;
                  enum cudaTextureReadMode    
                  readMode;
                  int                         sRGB;
                  float                       borderColor[4];
                  int                         normalizedCoords;
                  unsigned int                maxAnisotropy;
                  enum cudaTextureFilterMode  
                  mipmapFilterMode;
                  float                       mipmapLevelBias;
                  float                       minMipmapLevelClamp;
                  float                       maxMipmapLevelClamp;
              };
```
where  
* **`cudaTextureDesc::filterMode`**
`cudaTextureDesc::filterMode` specifies the filtering mode to be used when fetching from the texture. `cudaTextureFilterMode` is defined as:
```
    ‎        enum cudaTextureFilterMode {
                      cudaFilterModePoint  = 0,
                      cudaFilterModeLinear = 1
                  };
```
e.g. from [`./samples02/tex2dcuArray.cu`](https://github.com/ernestyalumni/CompPhys/blob/master/moreCUDA/samples02/tex2dcuArray.cu), and `simplePitchLinearTexture.cu` in `NVIDIA_CUDA-8.0_Samples/0_Simple`, indexed in [CUDA Runtime API Samples](http://docs.nvidia.com/cuda/cuda-samples/index.html#runtime-cudaapi), [Programming Interface, CUDA Toolkit Documentation](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#axzz4PoIffz3h)
```
texture<float,2,cudaReadModeElementType> texreference;
texture<float, 2, cudaReadModeElementType> texRefPL;
texture<float, 2, cudaReadModeElementType> texRefArray;

struct cudaTextureDesc texDesc;
memset(&texDesc,0, sizeof(texDesc));

texreference.filterMode=cudaFilterModePoint;

texRefPL.normalized = 1;
texRefPL.filterMode = cudaFilterModePoint;
texRefArray.normalized = 1;
texRefArray.filterMode = cudaFilterModePoint;

texDesc.addressMode[0]   = cudaAddressModeWrap;
texDesc.addressMode[1]   = cudaAddressModeWrap;
texDesc.filterMode       = cudaFilterModeLinear;
texDesc.readMode         = cudaReadModeElementType;
texDesc.normalizedCoords = 1;
```  
* **`cudaTextureDesc::addressMode`**  
`cudaTextureDesc::addressMode` specifies the addressing mode for each dimension of the texture data. `cudaTextureAddressMode` is defined as:
```
    ‎        enum cudaTextureAddressMode {
                      cudaAddressModeWrap   = 0,
                      cudaAddressModeClamp  = 1,
                      cudaAddressModeMirror = 2,
                      cudaAddressModeBorder = 3
                  };
```
This is ignored if `cudaResourceDesc::resType` is `cudaResourceTypeLinear`. Also, if `cudaTextureDesc::normalizedCoords` is set to zero, `cudaAddressModeWrap` and `cudaAddressModeMirror` won't be supported and will be switched to `cudaAddressModeClamp`.

e.g. from [`./samples02/tex2dcuArray.cu`](https://github.com/ernestyalumni/CompPhys/blob/master/moreCUDA/samples02/tex2dcuArray.cu), and `simplePitchLinearTexture.cu` in `NVIDIA_CUDA-8.0_Samples/0_Simple`, indexed in [CUDA Runtime API Samples](http://docs.nvidia.com/cuda/cuda-samples/index.html#runtime-cudaapi), [Programming Interface, CUDA Toolkit Documentation](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#axzz4PoIffz3h)
```
texture<float,2,cudaReadModeElementType> texreference;
texture<float, 2, cudaReadModeElementType> texRefPL;
texture<float, 2, cudaReadModeElementType> texRefArray;

struct cudaTextureDesc texDesc;
memset(&texDesc,0, sizeof(texDesc));

	// set texture address mode property
	// use cudaAddressModeClamp or cudaAddressModeWrap
texreference.addressMode[0]=cudaAddressModeWrap;
texreference.addressMode[1]=cudaAddressModeClamp;

texRefPL.addressMode[0] = cudaAddressModeWrap;
texRefPL.addressMode[1] = cudaAddressModeWrap;
texRefArray.addressMode[0] = cudaAddressModeWrap;
texRefArray.addressMode[1] = cudaAddressModeWrap;

texDesc.addressMode[0]   = cudaAddressModeWrap;
texDesc.addressMode[1]   = cudaAddressModeWrap;
texDesc.filterMode       = cudaFilterModeLinear;
texDesc.readMode         = cudaReadModeElementType;
texDesc.normalizedCoords = 1;
```
- **`cudaMemcpyToArray`**  
cf. [Memory Management, CUDA Toolkit 8.0 Documentation](http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1gcc65e278074cfe8f06aaa25788b7dc25)
```
__host__ ​cudaError_t cudaMemcpyToArray ( cudaArray_t dst,
	 	     		       	 size_t wOffset,
					 size_t hOffset,
					 const void* src, size_t count,
					 cudaMemcpyKind kind )
```  
Copies data between host and device.  

*Parameters*
`dst`  
	- Destination memory address   
`wOffset`  
        - Destination starting X offset  
    `hOffset`  
        - Destination starting Y offset   
    `src`    
        - Source memory address   
    `count`  
        - Size in bytes to copy  
    `kind`  
        - Type of transfer  

    *Returns*  
```
    cudaSuccess,
    cudaErrorInvalidValue,
    cudaErrorInvalidDevicePointer,
    cudaErrorInvalidMemcpyDirection
```  
*Description*

Copies count bytes from the memory area pointed to by src to the CUDA array dst starting at the upper left corner (wOffset, hOffset), where kind specifies the direction of the copy, and must be one of `cudaMemcpyHostToHost, cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice`, or `cudaMemcpyDefault`. Passing `cudaMemcpyDefault` is recommended, in which case the type of transfer is inferred from the pointer values. However, `cudaMemcpyDefault` is only allowed on systems that support unified virtual addressing.

Note:  
* Note that this function may also return error codes from previous, asynchronous launches.    
* This function exhibits synchronous behavior for most use cases.  
e.g. cf. [`./samples02/tex2dcuArray.cu`](https://github.com/ernestyalumni/CompPhys/blob/master/moreCUDA/samples02/tex2dcuArray.cu), `simplePitchLinearTexture.cu` in `NVIDIA_CUDA-8.0_Samples/0_Simple`
```
float* hmatrix
cudaArray* carray
const int bytes = sizeof(float)*size*size

// Set array size
const int nx = 2048;
const int ny = 2048;

// Host allocation and initialization
float *h_idata = (float *) malloc(sizeof(float) * nx * ny ) ; 

// Array intput data
cudaArray *d_idataArray;

// 0 is the wOffset, destination starting X offset
// 0 is the hOffset, destination starting Y offset
cudaMemcpyToArray(carray,0,0,hmatrix,bytes,cudaMemcpyHostToDevice);

cudaMemcpyToArray(d_idataArray,
		0,
		0,
		h_idata,
		nx * ny * sizeof(float),
		cudaMemcpyHostToDevice));
```
- **`memset`**
cf. [**`memset`**](http://www.cplusplus.com/reference/cstring/memset/), 


.
function
<cstring>
memset

void * memset ( void * ptr, int value, size_t num );

Fill block of memory
Sets the first num bytes of the block of memory pointed by ptr to the specified value (interpreted as an unsigned char).

Parameters

ptr
    Pointer to the block of memory to fill.
value
    Value to be set. The value is passed as an int, but the function fills the block of memory using the unsigned char conversion of this value.
num
    Number of bytes to be set to the value.
    size_t is an unsigned integral type.
e.g. [CUDA pro tip kepler texture objects](https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-kepler-texture-objects-improve-performance-and-flexibility/)
```
memset(&resDesc, 0, sizeof(resDesc));
```
- **`cudaCreateChannelDesc`**  
cf. [4.23. Texture Reference Management](http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TEXTURE.html#group__CUDART__TEXTURE_1g39df9e3b6edc41cd6f189d2109672ca5)
```  
__host__ ​cudaChannelFormatDesc cudaCreateChannelDesc ( int  x, int  y, int  z, int  w,
	 		       			       cudaChannelFormatKind f )
```  
Returns a channel descriptor using the specified format. 


*Parameters*  
    `x`  
        - X component  
    `y`  
        - Y component   
    `z`  
        - Z component   
    `w`  
        - W component    
    `f`   
        - Channel format   

*Returns*  
    Channel descriptor with format `f`   

*Description*  
    Returns a channel descriptor with format `f` and number of bits of each component `x, y, z`, and `w`. The `cudaChannelFormatDesc` is defined as:
```  
    ‎  struct cudaChannelFormatDesc {
              int x, y, z, w;
              enum cudaChannelFormatKind 
                      f;
            };
```  
where `cudaChannelFormatKind` is one of `cudaChannelFormatKindSigned`, `cudaChannelFormatKindUnsigned`, or `cudaChannelFormatKindFloat`. 

e.g. from [3.2.11.1.1. Texture Object API of CUDA Toolkit 8 Documentation](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#abstract), and also [`./samples02/simpletransform.cu`](https://github.com/ernestyalumni/CompPhys/blob/master/moreCUDA/samples02/simpletransform.cu)  
```
cudaChannelFormatDesc channelDesc =
		      cudaCreateChannelDesc(32, 0, 0, 0,
		      				cudaChannelFormatKindFloat);
```

### Implementation (of texture memory)

As a sanity check, I made sure that I can copy from host to device, and then onto texture memory the function

$f(x) = sin( 2\pi x) \cdot sin(2\pi y) \qquad \forall \, x, y \in [0,1]$

and then, after saving to `.csv` files, plot them with Python's `matplotlib` and displayed them in a jupyter notebook.  I did it here, in [`sinsin2dtex.ipynb`](https://github.com/ernestyalumni/CompPhys/blob/master/moreCUDA/samples02/sinsin2dtex.ipynb)


**If** you cannot write to texture memory, but only be able to write to surface memory, then I will implement in surface memory.  

## *Many* different `cudaMemcpy*`'s to try (I honestly don't know which one to use)
cf. [4.9 Memory Management, CUDA Runtime API Documentation](http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g6728eb7dc25f332f50bdb16a19620d3d)

There are *many* different `cudaMemcpy*`'s to try, from the CUDA Toolkit v8.0 Documentation.  I can't find any basic examples to illustrate each one, so I'll go one-by-one trying them.  
```
__host__ ​cudaError_t cudaMemcpy2D ( void* dst,
	 	     		    size_t dpitch,
				    const void* src,
				    size_t spitch,
				    size_t width,
				    size_t height,
				    cudaMemcpyKind kind )
```  
Copies data between host and device. 
```
__host__ ​cudaError_t cudaMemcpy2DArrayToArray ( cudaArray_t dst,
	 	     			      	size_t wOffsetDst,
						size_t hOffsetDst,
						cudaArray_const_t src,
						size_t wOffsetSrc,
						size_t hOffsetSrc,
						size_t width,
						size_t height,
						cudaMemcpyKind kind = cudaMemcpyDeviceToDevice )
```						
Copies data between host and device. 
```
__host__ ​ __device__ ​cudaError_t cudaMemcpy2DAsync ( void* dst,
	  	     		 		     size_t dpitch,
						     const void* src,
						     size_t spitch,
						     size_t width,
						     size_t height,
						     cudaMemcpyKind kind,
						     cudaStream_t stream = 0 )
```  						     
Copies data between host and device.   

```  
__host__ ​cudaError_t cudaMemcpy2DFromArray ( void* dst,
	 	     			     size_t dpitch,
					     cudaArray_const_t src,
					     size_t wOffset, size_t hOffset,
					     size_t width, size_t height, cudaMemcpyKind kind )  
```   

Copies data between host and device. 

```   
__host__ ​cudaError_t cudaMemcpy2DFromArrayAsync ( void* dst,
	 	     				  size_t dpitch,
						  cudaArray_const_t src,
						  size_t wOffset,
						  size_t hOffset,
						  size_t width,
						  size_t height,
						  cudaMemcpyKind kind, cudaStream_t stream = 0 )
```   

    Copies data between host and device. 

```
__host__ ​cudaError_t cudaMemcpy2DToArray ( cudaArray_t dst,
	 	     			   size_t wOffset, size_t hOffset,
					   const void* src, size_t spitch,
					   size_t width, size_t height,
					   cudaMemcpyKind kind )
```  

Copies data between host and device. 

```  
__host__ ​cudaError_t cudaMemcpy2DToArrayAsync ( cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream = 0 )
```  

Copies data between host and device. 

```  
__host__ ​cudaError_t cudaMemcpy3D ( const cudaMemcpy3DParms* p )  
```  

Copies data between 3D objects. 

```  
__host__ ​ __device__ ​cudaError_t cudaMemcpy3DAsync ( const cudaMemcpy3DParms* p,
	  	     		 		     cudaStream_t stream = 0 )
```  

Copies data between 3D objects.   

```
__host__ ​cudaError_t cudaMemcpy3DPeer ( const cudaMemcpy3DPeerParms* p )
```  

Copies memory between devices. 

```
__host__ ​cudaError_t cudaMemcpy3DPeerAsync ( const cudaMemcpy3DPeerParms* p,
	 	     			     cudaStream_t stream = 0 )

```  
Copies memory between devices asynchronously. 
  
```  
__host__ ​cudaError_t cudaMemcpyArrayToArray ( cudaArray_t dst,
	 	     			      size_t wOffsetDst,
					      size_t hOffsetDst,
					      cudaArray_const_t src,
					      size_t wOffsetSrc,
					      size_t hOffsetSrc,
					      size_t count,
					      cudaMemcpyKind kind = cudaMemcpyDeviceToDevice )

```  					      

Copies data between host and device. 

```  
__host__ ​cudaError_t cudaMemcpyFromArray ( void* dst,
	 	     			   cudaArray_const_t src,
					   size_t wOffset,
					   size_t hOffset,
					   size_t count,
					   cudaMemcpyKind kind )
```   

Copies data between host and device. 


```  
__host__ ​cudaError_t cudaMemcpyToArray ( cudaArray_t dst,
	 	     		       	 size_t wOffset,
					 size_t hOffset,
					 const void* src,
					 size_t count,
					 cudaMemcpyKind kind )
```  					 

Copies data between host and device. 

```
__host__ ​cudaError_t cudaMemcpyToArrayAsync ( cudaArray_t dst,
	 	     			      size_t wOffset,
					      size_t hOffset,
					      const void* src,
					      size_t count,
					      cudaMemcpyKind kind,
					      cudaStream_t stream = 0 )  
```   

Copies data between host and device. 

From [`./samples02/simplelinear2dtex.cu`](https://github.com/ernestyalumni/CompPhys/blob/master/moreCUDA/samples02/simplelinear2dtex.cu) is an example of `cudaMemcpyToArray`:  

```  
cudaMemcpyToArray(cuArray,0,0,(grid2d.rho).data(), sizeof(float)*grid2d.NFLAT(),
							cudaMemcpyHostToDevice)
```  



## Surface Memory

*EY : 20161116* I was having trouble reading and writing with *Texture Object* by modifying the associated array on the device.  See [`./samples02/texdynamics/`](https://github.com/ernestyalumni/CompPhys/tree/master/moreCUDA/samples02/texdynamics), which is what I have so far.  

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

cf. [CUDA Runtime API Samples](http://docs.nvidia.com/cuda/cuda-samples/#runtime-cudaapi)

*Table 5. CUDA Runtime API and Associated Samples*

| CUDA Runtime API |	Samples  |
| :--------------- | :---------: |
| `cudaBindSurfaceToArray` | Simple Surface Write |

* Added `2_Graphics/bindlessTexture` - demonstrates use of `cudaSurfaceObject, cudaTextureObject`, and MipMap support in CUDA. Requires Compute Capability 3.0 or higher.

Take a look at [3.2.11.2.1. Surface Object API](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#surface-object-api)

Consider what this means
```
// Allocate CUDA arrays in device memory
    cudaChannelFormatDesc channelDesc =
             cudaCreateChannelDesc(8, 8, 8, 8,
                                   cudaChannelFormatKindUnsigned);  

    cudaArray* cuInputArray;
    cudaMallocArray(&cuInputArray, &channelDesc, width, height,
                    cudaArraySurfaceLoadStore);
    cudaArray* cuOutputArray;
    cudaMallocArray(&cuOutputArray, &channelDesc, width, height,
                    cudaArraySurfaceLoadStore);
```  

cf. [3.2.11.2.2. Surface Reference API](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#surface-reference-api)

Unlike texture memory, surface memory uses byte addressing. This means that the x-coordinate used to access a texture element via texture functions needs to be multiplied by the byte size of the element to access the same element via a surface function. For example, the element at texture coordinate x of a one-dimensional floating-point CUDA array bound to a texture reference `texRef` and a surface reference `surfRef` is read using `tex1d(texRef, x)` via `texRef`, but `surf1Dread(surfRef, 4*x)` via `surfRef`. Similarly, the element at texture coordinate x and y of a two-dimensional floating-point CUDA array bound to a texture reference texRef and a surface reference surfRef is accessed using `tex2d(texRef, x, y)` via `texRef`, but `surf2Dread(surfRef, 4*x, y)` via `surfRef` (the byte offset of the y-coordinate is internally calculated from the underlying line pitch of the CUDA array).

The following code sample applies some simple transformation kernel to a texture.
```  
// 2D surfaces
surface<void, 2> inputSurfRef;
surface<void, 2> outputSurfRef;
            
// Simple copy kernel
__global__ void copyKernel(int width, int height) 
{
    // Calculate surface coordinates
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        uchar4 data;
        // Read from input surface
        surf2Dread(&data,  inputSurfRef, x * 4, y);
        // Write to output surface
        surf2Dwrite(data, outputSurfRef, x * 4, y);
    }  
```  

cf. [B.9.1. Surface Object API](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#surface-object-api-appendix)

B.9.1.1. surf1Dread()

template<class T>
T surf1Dread(cudaSurfaceObject_t surfObj, int x,
               boundaryMode = cudaBoundaryModeTrap);

reads the CUDA array specified by the one-dimensional surface object surfObj using coordinate x.
B.9.1.2. surf1Dwrite

template<class T>
void surf1Dwrite(T data,
                  cudaSurfaceObject_t surfObj,
                  int x,
                  boundaryMode = cudaBoundaryModeTrap);

writes value data to the CUDA array specified by the one-dimensional surface object surfObj at coordinate x.  
[B.9.1.3. surf2Dread()](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#surf2dread-object)
```  
template<class T>
T surf2Dread(cudaSurfaceObject_t surfObj,
              int x, int y,
              boundaryMode = cudaBoundaryModeTrap);
template<class T>
void surf2Dread(T* data,
                 cudaSurfaceObject_t surfObj,
                 int x, int y,
                 boundaryMode = cudaBoundaryModeTrap);  
```  
reads the CUDA array specified by the two-dimensional surface object surfObj using coordinates x and y.
[B.9.1.4. surf2Dwrite()](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#surf2dwrite-object)
```  
template<class T>
void surf2Dwrite(T data,
                  cudaSurfaceObject_t surfObj,
                  int x, int y,
                  boundaryMode = cudaBoundaryModeTrap);  
```  
writes value data to the CUDA array specified by the two-dimensional surface object surfObj at coordinate x and y.  
[B.9.1.5. surf3Dread()](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#surf3dread-object)

template<class T>
T surf3Dread(cudaSurfaceObject_t surfObj,
              int x, int y, int z,
              boundaryMode = cudaBoundaryModeTrap);
template<class T>
void surf3Dread(T* data,
                 cudaSurfaceObject_t surfObj,
                 int x, int y, int z,
                 boundaryMode = cudaBoundaryModeTrap);

reads the CUDA array specified by the three-dimensional surface object surfObj using coordinates x, y, and z.
B.9.1.6. surf3Dwrite()

template<class T>
void surf3Dwrite(T data,
                  cudaSurfaceObject_t surfObj,
                  int x, int y, int z,
                  boundaryMode = cudaBoundaryModeTrap);

writes value data to the CUDA array specified by the three-dimensional object surfObj at coordinate x, y, and z.
B.9.1.7. surf1DLayeredread()

template<class T>
T surf1DLayeredread(
                 cudaSurfaceObject_t surfObj,
                 int x, int layer,
                 boundaryMode = cudaBoundaryModeTrap);
template<class T>
void surf1DLayeredread(T data,
                 cudaSurfaceObject_t surfObj,
                 int x, int layer,
                 boundaryMode = cudaBoundaryModeTrap);

reads the CUDA array specified by the one-dimensional layered surface object surfObj using coordinate x and index layer.
B.9.1.8. surf1DLayeredwrite()

template<class Type>
void surf1DLayeredwrite(T data,
                 cudaSurfaceObject_t surfObj,
                 int x, int layer,
                 boundaryMode = cudaBoundaryModeTrap);

writes value data to the CUDA array specified by the two-dimensional layered surface object surfObj at coordinate x and index layer.
B.9.1.9. surf2DLayeredread()

template<class T>
T surf2DLayeredread(
                 cudaSurfaceObject_t surfObj,
                 int x, int y, int layer,
                 boundaryMode = cudaBoundaryModeTrap);
template<class T>
void surf2DLayeredread(T data,
                         cudaSurfaceObject_t surfObj,
                         int x, int y, int layer,	
                         boundaryMode = cudaBoundaryModeTrap);

reads the CUDA array specified by the two-dimensional layered surface object surfObj using coordinate x and y, and index layer.
B.9.1.10. surf2DLayeredwrite()

template<class T>
void surf2DLayeredwrite(T data,
                          cudaSurfaceObject_t surfObj,
                          int x, int y, int layer,
                          boundaryMode = cudaBoundaryModeTrap);

writes value data to the CUDA array specified by the one-dimensional layered surface object surfObj at coordinate x and y, and index layer.
B.9.1.11. surfCubemapread()

template<class T>
T surfCubemapread(
                 cudaSurfaceObject_t surfObj,
                 int x, int y, int face,
                 boundaryMode = cudaBoundaryModeTrap);
template<class T>
void surfCubemapread(T data,
                 cudaSurfaceObject_t surfObj,
                 int x, int y, int face,
                 boundaryMode = cudaBoundaryModeTrap);

reads the CUDA array specified by the cubemap surface object surfObj using coordinate x and y, and face index face.
B.9.1.12. surfCubemapwrite()

template<class T>
void surfCubemapwrite(T data,
                 cudaSurfaceObject_t surfObj,
                 int x, int y, int face,
                 boundaryMode = cudaBoundaryModeTrap);

writes value data to the CUDA array specified by the cubemap object surfObj at coordinate x and y, and face index face.
B.9.1.13. surfCubemapLayeredread()

template<class T>
T surfCubemapLayeredread(
             cudaSurfaceObject_t surfObj,
             int x, int y, int layerFace,
             boundaryMode = cudaBoundaryModeTrap);
template<class T>
void surfCubemapLayeredread(T data,
             cudaSurfaceObject_t surfObj,
             int x, int y, int layerFace,
             boundaryMode = cudaBoundaryModeTrap);

reads the CUDA array specified by the cubemap layered surface object surfObj using coordinate x and y, and index layerFace.
B.9.1.14. surfCubemapLayeredwrite()
```  
template<class T>
void surfCubemapLayeredwrite(T data,
             cudaSurfaceObject_t surfObj,
             int x, int y, int layerFace,
             boundaryMode = cudaBoundaryModeTrap);
```  
writes value data to the CUDA array specified by the cubemap layered object surfObj at coordinate x and y, and index layerFace.


[stackoverflow: CUDA textures and clamping](http://stackoverflow.com/questions/5757294/cuda-textures-and-clamping)

From [pQB](http://stackoverflow.com/users/714967/pqb)'s answer, "We can also clamp the boundary and trap it (make kernel fail) what is the default when using the surface memory."


[4.31. Data types used by CUDA Runtime](http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#axzz4QD8orpjR)

```  
enum cudaSurfaceBoundaryMode
```  
CUDA Surface boundary modes

**Values**
```  
    cudaBoundaryModeZero = 0
```    
Zero boundary mode 
```  
cudaBoundaryModeClamp = 1
```  
Clamp boundary mode 
```  
cudaBoundaryModeTrap = 2
```  
Trap boundary mode 

So the default value for the `boundaryMode` argument in `surf2Dread`, `surf2Dwrite` and other `surf*` is `cudaBoundaryModeTrap, which fails when access is outside the array.  




## Shared Memory  

[Using Shared Memory in CUDA C/C++](https://devblogs.nvidia.com/parallelforall/using-shared-memory-cuda-cc/) by [Mark Harris](https://devblogs.nvidia.com/parallelforall/author/mharris/)

## `cudaMallocArray` and associated examples (in NVIDIA CUDA 8.0 Samples)
cf.  [4.9 Memory Management, CUDA Runtime API Documentation](http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g6728eb7dc25f332f50bdb16a19620d3d)
```
__host__ ​cudaError_t cudaMallocArray ( cudaArray_t* array,
	 	     		       const cudaChannelFormatDesc* desc,
				       size_t width, size_t height = 0,
				       unsigned int  flags = 0 )
```
    Allocate an array on the device.

*Parameters*

    `array`
        - Pointer to allocated array in device memory 
    `desc`
        - Requested channel format 
    `width`
        - Requested array allocation width 
    `height`
        - Requested array allocation height 
    `flags`
        - Requested properties of allocated array





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

## `thrust` and *useful links* for *`thrust`*

[icuda hands-on introduction to CUDA programming](http://istar.cse.cuhk.edu.hk/icuda/) - very good hands-on introduction to CUDA programming, chock full of executable and thorough examples.  It's also well-formatted (the typography is even nice to look at).  I've put on github the examples I've typed up.  

http://istar.cse.cuhk.edu.hk/icuda/

# CUB

> > "CUB, on the other hand, is a production-quality library whose sources are complicated by support for every version of CUDA architecture, and is validated by an extensive suite of regression tests."  - [(7) How is CUB different than Thrust and Modern GPU?](https://nvlabs.github.io/cub/index.html#sec4sec1)  

