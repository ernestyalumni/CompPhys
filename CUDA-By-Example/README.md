# `CUDA-By-Example` CUDA by Example
#### cf. Jason Sanders, Edward Kandrot. **CUDA by Example: An Introduction to General-Purpose GPU Programming**

I also cloned in this same repository the github repository from [jiekebo](https://github.com/jiekebo) for [CUDA-By-Example](https://github.com/jiekebo/CUDA-By-Example), and this is a general observation: it may help to first search on github for the code you seek because it seems likely that someone already wrote it.

## Note on examples out of CUDA by Example

It seems that a header file `book.h` out of the `common` subdirectory is needed to run the scripts.  I've tried to write my own scripts without needing the `book.h` header.  However, it's found in jiekebo's repository and in this repository, in the subdirectory (from here) `CUDA-By-Example`.  I don't know the rationale behind `book.h` or why the authors made you need it for their examples (as I am reading the book, it's not explained (!!!)).

## Dictionary between files on this github subdirectory to code in **CUDA By Example**, *Sanders and Kandrot*

I'm also looking at Bjarne Stroustrup's **A Tour of C++** (2013) Addison-Wesley, and Stroustrup (2013) refers to this text.  

| filename       |   pp.  | (Sub)Section             | Description                  |
| -------------- | :----: | :--------------------:   | :--------------------------: |
| helloworld.c   | 22     | 3.2 A First Program      | Hello world in C             |
| helloworld.cpp | 22     | 3.2 A First Program      | Hello world in C++; also in pp. 3, Section 1.3 Hello World! of Stroustrup (2013) |
| helloworld.cu  | 23     | 3.2.2 A Kernel Call      | Hello world in CUDA C        |
| [add-pass.cu](https://github.com/ernestyalumni/CompPhys/blob/master/CUDA-By-Example/add-pass.cu) |  25  | 3.2.3 Passing Parameters | example of using cudaMalloc, cudaMemcpy, cudaFree |
| [add-passb.cu](https://github.com/ernestyalumni/CompPhys/blob/master/CUDA-By-Example/add-passb.cu) |  25  | 3.2.3 Passing Parameters | example of using cudaMalloc, cudaMemcpy, cudaFree, with my changes, using my own errors.h header file |
| query.cu       | 27     | 3.3 Querying Devices     | query each device and report the properties of each |
| queryb.cu      | 32     | 3.3 Querying Devices     | query each device and report the properties of each |
| [cpuvecsum.c](https://github.com/ernestyalumni/CompPhys/blob/master/CUDA-By-Example/cpuvecsum.c) |  40  | 4.2.1 Summing Vectors    | Sum vectors as an C array    |
| [cpuvecsum.cpp](https://github.com/ernestyalumni/CompPhys/blob/master/CUDA-By-Example/cpuvecsum.cpp) |  40  | 4.2.1 Summing Vectors    | Sum vectors as an C++ array; see pp. 9-12 of 1.8 Pointers, Arrays, and References of Stroustrup (2013)     |
| [gpuvecsum.cu](https://github.com/ernestyalumni/CompPhys/blob/master/CUDA-By-Example/gpuvecsum.cu) | 41 | 4.2.1 Summing Vectors    | Sum arrays as vectors in GPU |
| [gpujulia.cu](https://github.com/ernestyalumni/CompPhys/blob/master/CUDA-By-Example/gpujulia.cu) | 46-57 | 4.2.2 A Fun Example    | Julia Sets! |
| [gpujuliab.cu](https://github.com/ernestyalumni/CompPhys/blob/master/CUDA-By-Example/gpujuliab.cu) | 46-57 | 4.2.2 A Fun Example    | Julia Sets! My own version, where you can set the scale factor when calling the executable (argv) |
| gpuvecsumredux.cu  | 61-62 | 5.2.1 Vector Sums: Redux | GPU Vector Sums using Threads|
| gpuvecsumredux2.cu  | 67-69 | 5.2.1 Vector Sums: Redux | GPU Vector Sums using Blocks and Threads|
| gpuripple.cu  | 69-74 | 5.2.2 GPU Ripple Using Threads | Animated standing wave |
| dotprod.cu    | 69-74 | 5.3.1 Dot product | illustrations sharing between threads in a block, novel dot product summation scheme, using `__shared__` |
| dotprodcpp.cu    | 69-74 | 5.3.1 Dot product | Same as `dotprod.cu` but I attempt to implement C++11 coding methods, such as `new` and `delete`, as opposed to `malloc` and `free` |
| sharedbitmap.cu | 90-93 | 5.3.2 Shared Memory Bitmap | demonstrates usage of `__syncthreads();` |
| raytrace.cu     | 99-103 | 6.2.2 Ray Tracing on the GPU | ray tracing randomly placed, but *fixed* sphere, without using **constant memory** |
| rayconst.cu     | 104-113 | 6.2.3 Ray Tracing with Constant Memory | ray tracing randomly placed, but *fixed* sphere, using **constant memory** |
| simpleheat.cu    | 117-125 | 7.3.1 Simple Heating Model, 7.3.2 Computing Temperature Updates | using global GPU memory for simple heating model, no texture memory involved |
| heattexture1.cu  | 125-131 | 7.3.4 Using Texture Memory | Demonstrates 1-dim. texture memory |
| heattexture2d.cu | 131-137 | 7.3.5 Using Two-Dimensional Texture Memory | Demonstrates 2-dim. texture memory |
| graphicsinterop.cu | 140-147 | 8.2 Graphics Interoperation | pedagogical example of graphics interoperation between CUDA C/C++ and OpenGL |
| gpuripplereudx.cu  | 152-154 | 8.3.2 GPU Ripple Redux | gpuripple, but rendering on the GPU; notice how simpler the code is since you *don't* have to `cudaMemcpy` the bitmap back to the host (CPU) |
| simpleheatredux.cu | 154-160 | 8.4 Heat Transfer with Graphics Interop | Heat transfer, no using texture memory, using GPU rendering; roughly from 8.1 ms to 6.9 ms time needed to render frames performance *boost* on NVIDIA GeForce 980ti |


## Awesome gallery of results out of CUDA C/C++, CUDA By Examples

Pretty graphics and animation keeps (me, at least) one motivated.

From [`gpujulia.cu`](https://github.com/ernestyalumni/CompPhys/blob/master/CUDA-By-Example/gpujulia.cu),

<img src=https://github.com/ernestyalumni/CompPhys/blob/master/CUDA-By-Example/imgs/gpujuliaScreenshot%20from%202016-06-09%2001-02-14.png width=400 height=400 />

![From `gpujulia.cu`](https://github.com/ernestyalumni/CompPhys/blob/master/CUDA-By-Example/imgs/gpujuliaScreenshot%20from%202016-06-09%2001-02-14.png "From gpujulia.cu")

From [`gpujuliab.cu`](https://github.com/ernestyalumni/CompPhys/blob/master/CUDA-By-Example/gpujuliab.cu), which you (and I) can open and change around the constants to obtain different Julia sets:

<img src=https://github.com/ernestyalumni/CompPhys/blob/master/CUDA-By-Example/imgs/gpujuliabScreenshot%20from%202016-06-09%2003-06-37.png width=400 height=400 />

From [`gpuripple.cu`](https://github.com/ernestyalumni/CompPhys/blob/master/CUDA-By-Example/gpuripple.cu), I'll try to embed a Twitter "tweet" of the video:
<blockquote class="twitter-video" data-lang="en"><p lang="pl" dir="ltr"><a href="https://twitter.com/GPUComputing">@GPUComputing</a> CUDA C program for waves on <a href="https://twitter.com/NVIDIAGeForce">@NVIDIAGeForce</a> GTX980ti on <a href="https://twitter.com/TitanusComputer">@TitanusComputer</a> <a href="https://t.co/i2PAzkTXia">https://t.co/i2PAzkTXia</a> <a href="https://t.co/FcaMNfx7SK">pic.twitter.com/FcaMNfx7SK</a></p>&mdash; Ernest Yeung (@ernestyalumni) <a href="https://twitter.com/ernestyalumni/status/741014181323759616">June 9, 2016</a></blockquote>
<script async src="//platform.twitter.com/widgets.js" charset="utf-8"></script>


From [`sharedbitmap.cu`](https://github.com/ernestyalumni/CompPhys/blob/master/CUDA-By-Example/sharedbitmap.cu), which demonstrates correct usage of `__syncthreads()` of CUDA C:

<img src=https://github.com/ernestyalumni/CompPhys/blob/master/CUDA-By-Example/imgs/sharedbitmapScreenshot%20from%202016-06-09%2017-23-53.png width=400 height=400 />

From [`raytrace.cu`](https://github.com/ernestyalumni/CompPhys/blob/master/CUDA-By-Example/raytrace.cu), 40 spheres, with `Time to generate: 2.9 ms`,

<img src=https://github.com/ernestyalumni/CompPhys/blob/master/CUDA-By-Example/imgs/raytraceScreenshot%20from%202016-06-10%2001-25-56.png width=400 height=400 />

From [`rayconst.cu`](https://github.com/ernestyalumni/CompPhys/blob/master/CUDA-By-Example/rayconst.cu), 50 spheres, with `Time to generate: 0.7 ms`,

<img src=https://github.com/ernestyalumni/CompPhys/blob/master/CUDA-By-Example/imgs/rayconstScreenshot%20from%202016-06-10%2002-00-25.png width=400 height=400 />

From [`graphicsinterop.cu`](https://github.com/ernestyalumni/CompPhys/blob/master/CUDA-By-Example/graphicsinterop.cu), 

<img src=https://github.com/ernestyalumni/CompPhys/blob/master/CUDA-By-Example/imgs/graphicsinteropScreenshot%20from%202016-06-15%2016-02-56.png width=400 height=400 />

## Compiling and running these programs in CUDA C/C++

Most of the time, these commands will work in compiling (`nvcc`) and running (`./a.out`) these programs (in this repository subdirectory `CUDA-By-Example`):  

```
nvcc filename.cu  
./a.out  
```  
e.g.  
```
nvcc helloworldkernel.cu
./a.out  
```

However, sometimes you'll need `glut`.  Look at the program's `include`'s and header files.

By the way, on that note, in my experience with Fedora Linux 23, I found that `freeglut` RPM didn't have the header file `glut.h` I needed.  So then I had to install this RPM, `freeglut-devel`.  Then it had all the header files I needed.  cf. [How to Setup OpenGL (installing glut) on Linux and Compile Programs](http://sa-os.blogspot.com/2010/01/how-to-setup-opengl-on-linux-and.html).  I did this with `dnf list freeglut-devel` and `dnf install freeglut-devel.x86_64`.

So for instance, for `gpujulia.cu`, you'll need to compile it with these flags for `nvcc` (`-l` is the short name for the flag for library, to specify libraries to be used in linking stage, without the library file extension; cf. [3.2.2. File and path specifications](http://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#axzz4B3AvO2IG):  

```
nvcc -lglut -lGL gpujulia.cu
```


### Querying Devices
cf. 3.3 Querying Devices of Sanders and Kandrot (2010).

Take a look at `query.cu`.  

It appears that `cudaDeviceCount` obtains the total number of devices.  For example, in my case, it is 1, so that `count == 1`.  `cudaGetDeviceProperties` retrieves the properties of a device (specified by that count number).  One of those properties is its name, evoked by `.name`.  In my case, for example, after compiling `query.cu` by doing `nvcc query.cu`, then running the executable, by doing `./a.out`, then I obtain:
```
Name: GeForce GTX 980 Ti
```

Then `queryb.cu` has further properties that you "evoke" by various "methods" of the class, e.g. `.name`, `.major`, `.minor`, etc.

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

   --- Other Information for device 0 ---
Max. 3D textures dimensions: (4096, 4096, 4096) 
```

## On Graphics interoperability, i.e. CUDA C/C++ + OpenGL

I myself found the topic of "Graphics Interoperability" (cf. Ch. 8 of Sanders and Kandrot) to be challenging.  I'll rewrite/step through the code from [`graphicsinterop.cu`](https://github.com/ernestyalumni/CompPhys/blob/master/CUDA-By-Example/graphicsinterop.cu) here:

```
// global variables for same buffer
Gluint bufferObj;
cudaGraphicsResource *resource;
```
The `global` variables `bufferObj`, `resource` stores different "handles" to the same buffer, i.e. OpenGL and CUDA C/C++ will each have different "names" for the same buffer.


### Setting the CUDA device to play with OpenGL driver
This is done at the start; in [`graphicsinterop.cu`](https://github.com/ernestyalumni/CompPhys/blob/master/CUDA-By-Example/graphicsinterop.cu), it was done in `main`

We want to select the CUDA device to run the application.  We need to know the CUDA device ID so we can tell CUDA runtime that we intend to use the device for CUDA **AND** OpenGL.  
```
int main( int argc, char **argv) {
	cudaDeviceProp prop;
	int dev;

	memset( &prop, 0, sizeof( cudaDeviceProp ) );
	prop.major = 1; // major version set to 1
	prop.minor = 0; // minor version set to 0 
	
	cudaChooseDevice( &dev, &prop ); 

	cudaGLSetGLDevice( dev );
```  
`prop.major = 1` sets the major version to 1.  
`prop.minor = 0` sets the minor version to 0.  
These versions refer to compute versions.  (EY 20160616: I want to try for compute version 2).  

`cudaChooseDevice( &dev, &prop)` instructs the runtime to select GPU that satisfies constraints specified by `cudaDeviceProp`'s `prop`.  

We've now prepared our CUDA runtime to play nicely with OpenGL driver with `cudaGLSetGLDevice`.  

### Initialize OpenGL driver by calling GL Utility Toolkit (GLUT)

These GLUT calls need to be made before other GL calls.  

Note that we're still in `main` (for this *particular* example):
```
	glutInit( &argc, argv );
	glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGBA)
	glutInitWindowSize(width, height );
	glutCreateWindow( "bitmap" );
```
A note on `glutInit` - "some versions of GLUT have a bu that cause applications to crash when zero arguments are given" (Sanders and Kandrot (2011)).  So, trick GLUT into thinking we're passing an argument (if you're not going to use `&argc, argv`):
```
	int soo = 1;
	char *foo = "name" ;
	glutInit( &soo, &foo);
```  

### Instruct CUDA runtime to map shared resource, and then, by requesting pointer to mapped resource; the actual address in device memory of the *buffer*

We want to read from and write to this buffer, which OpenGL calls refer with handle `bufferObj`, and which CUDA runtime calls refer to with pointer `resource`.  

So we want the actual address in device memory.  

Achieve this by instructing CUDA runtime to map shared resource and then by requesting a pointer to the mapped resource.  

```
uchar4* devPtr;
size_t size; // unsigned integer type, to represent size of an object
cudaGraphicsMapResources( 1, &resource, NULL);
cudaGraphicsResourceGetMappedPointer( (void**)&devPtr,
	&size, resource);
```  
We can then use `devPtr` as we would any device pointer, except that the data can also be used by OpenGL as a pixel source. (Sanders and Kandrot, 2011).  




