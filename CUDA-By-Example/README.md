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