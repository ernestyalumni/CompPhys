## Multiple shared memory arrays/variables in CUDA  

cf. [SOLVED- Shared memory variable declaration`devtalk.nvidia.com`](https://devtalk.nvidia.com/default/topic/983672/-solved-shared-memory-variable-declaration/)  
cf. [B.2.3.`__shared__` in CUDA Toolkit Documentation v9](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared)  

Layout of arrays/variables in the array must be *explicitly managed* (by you, programmer) through *offsets*.  You can't have separate, multiple, in shared memory, variables/arrays explicitly. 
