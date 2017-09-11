# Notes, Solutions, and code for Cheng, Grossman, McKercher **Professional CUDA C Programming**.  

cf. John Cheng, Max Grossman, Ty McKercher. **Professional CUDA C Programming.** 1st Edition. *Wrox*; 1 edition (September 9, 2014). ISBN-13: 978-1118739327.

## Table of Contents  

| codename         | directory      | Chapter; Section or pp. in Cheng, Grossman, McKercher (2014) | Description             |
| ---------------- | -------------- | ---------------- | :----------------------: |  
| `commonmultiplestreams.cu` | `./` | Ch. 6 Streams and Concurrency; pp. 271 | Common pattern for dispatching CUDA operations to multiple streams; cf. [CUDA Programming Guide, 3.2.5. Asynchronous Concurrent Execution](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-concurrent-execution) |  


## CUDA Execution Model, Ch. 3 of Cheng, Grossman, McKercher (2014)

When kernel grid is launched, thread blocks of that kernel grid are distributed among available SMs for execution.  Once scheduled on SM, threads of a thread block execute concurrently only on that assigned SM.  

CUDA employs *Single Instruction Multiple Thread (SIMT)* architecture to manage and execute threads in groups of 32 called *warps*.  

All threads in warp execute same instruction at the same time.  Each thread has its own instruction address counter and register state, and carries out the current instruction on its own data.  Each SM partitions the thread blocks assigned to it into 32-thread warps that it then schedules for execution on available hardware resources.  

SIMT allows multiple threads in same warp to execute independently.  Even though all threads in a warps start together at the same program address, it's possible for individual threads to have different behavior.   

### Understanding the Nature of Warp Execution; Warps and Thread Blocks  

Once a thread block is scheduled to an SM, threads in thread block are further partitioned into warps.  A warp consists of 32 consecutive threads and all threads in a warp are executed in SIMT, i.e. all threads execute same instruction, and each thread carries out operation on its own private data.  

#### Warp Divergence  

For example, consider the following statement:  
```  
if (cond) {
    ...   
} else {
    ...
}
```  

Suppose for 16 threads in a warp executing this code, `cond` is `true`, but for other 16 `cond` is `false`.  Then half of the warp will need to execute the instructions in the `if` block, and the other half will need to execute the instructions in the `else` block.  Threads in the same warp executing different instructions is referred to as **warp divergence.**  

If threads of a warp diverge, the warp serially executes each branch path, disabling threads that do not take that path.  

e.g. All threads within a warp must take both branches of the `if ... then` statement.  If condition is `true` for a thread, it executes the `if` clause; otherwise, the thread stalls while waiting for that execution to complete.  

## Streams and Concurrency, Ch. 6 of Cheng, Grossman, McKercher (2014)

When performing an asynchronous data trasfer, you must use pinned (or non-pageable) host memory.  Pinned memory can be allocated using either `cudaMallocHost` or `cudaHostAlloc`:  

```  
cudaError_t cudaMallocHost(void **ptr, size_t size);
cudaError_t cudaHostAlloc(void **pHost, size_t size,unsigned int flags);
```  


cf. pp. 271, Introducing Streams and Events section,

Common pattern for dispatching CUDA operations to multiple streams:  

``` 
for (int i=0; i< nStreams; i++) {
    int offset = i*bytesPerStream;
    cudaMemcpyAsync(&d_a[offset], &a[offset], bytePerStream, streams[i]);
    kernel<<<grad,block,0,streams[i]>>>(&d_a[offset]);
    cudaMemcpyAsync(&a[offset], &d_a[offset], bytesPerStream, streams[i]); 
}

for (int i=0; i<nStreams;i++) {
    cudaStreamSynchronize(streams[i]);
}
``` 
