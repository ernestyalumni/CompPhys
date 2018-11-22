# on *Vector loads for higher memory access efficiency*  

cf. [CUDA Pro Tip: Increase Performance with Vectorized Memory Access, by Justin Luitjens](https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-increase-performance-with-vectorized-memory-access/)   

Many CUDA kernels are bandwidth bound.  
In new hardware, there's an increasing ratio of flops to bandwidth.  So the bottleneck is bandwidth.  
Use vector loads and stores to increase bandwidth utilization, while decreasing number of executed instructions.  

## inspect assembly in CUDA with [`cuobjdump`](http://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#cuobjdump)  

The SASS for the body of the scalar copy kernel, originally given by Luitjens, is the following:
```  
/*0058*/ IMAD R6.CC, R0, R9, c[0x0][0x140]                
/*0060*/ IMAD.HI.X R7, R0, R9, c[0x0][0x144]              
/*0068*/ IMAD R4.CC, R0, R9, c[0x0][0x148]               
/*0070*/ LD.E R2, [R6]                                   
/*0078*/ IMAD.HI.X R5, R0, R9, c[0x0][0x14c]              
/*0090*/ ST.E [R4], R2
```  
Here, total of 6 instructions associated with copy operation:  
- 4 `IMAD` instructions compute load and store addresses  
- `LD.E`, `ST.E` load and store 32 bits from those addresses  


I obtained the following:  

```  
Fatbin elf code:
================
arch = sm_30
code version = [1,7]
producer = cuda
host = linux
compile_size = 64bit

	code for sm_30
		Function : _Z25device_copy_scalar_kernelPiS_i
	.headerflags    @"EF_CUDA_SM30 EF_CUDA_PTX_SM(EF_CUDA_SM30)"
                                                                                /* 0x2202e2c282823307 */
        /*0008*/                   MOV R1, c[0x0][0x44];                        /* 0x2800400110005de4 */
        /*0010*/                   S2R R0, SR_CTAID.X;                          /* 0x2c00000094001c04 */
        /*0018*/                   S2R R3, SR_TID.X;                            /* 0x2c0000008400dc04 */
        /*0020*/                   IMAD R0, R0, c[0x0][0x28], R3;               /* 0x20064000a0001ca3 */
        /*0028*/                   ISETP.GE.AND P0, PT, R0, c[0x0][0x150], PT;  /* 0x1b0e40054001dc23 */
        /*0030*/               @P0 EXIT;                                        /* 0x80000000000001e7 */
        /*0038*/                   ISCADD R2.CC, R0, c[0x0][0x140], 0x2;        /* 0x4001400500009c43 */
                                                                                /* 0x228232c04282b047 */
        /*0048*/                   MOV32I R5, 0x4;                              /* 0x1800000010015de2 */
        /*0050*/                   MOV R6, c[0x0][0x34];                        /* 0x28004000d0019de4 */
        /*0058*/                   IMAD.HI.X R3, R0, R5, c[0x0][0x144];         /* 0x208a80051000dce3 */
        /*0060*/                   LD.E R2, [R2];                               /* 0x8400000000209c85 */
        /*0068*/                   ISCADD R4.CC, R0, c[0x0][0x148], 0x2;        /* 0x4001400520011c43 */
        /*0070*/                   IMAD.HI.X R5, R0, R5, c[0x0][0x14c];         /* 0x208a800530015ce3 */
        /*0078*/                   IMAD R0, R6, c[0x0][0x28], R0;               /* 0x20004000a0601ca3 */
                                                                                /* 0x20000002f2e043f7 */
        /*0088*/                   ISETP.LT.AND P0, PT, R0, c[0x0][0x150], PT;  /* 0x188e40054001dc23 */
        /*0090*/                   ST.E [R4], R2;                               /* 0x9400000000409c85 */
        /*0098*/               @P0 BRA 0x38;                                    /* 0x4003fffe600001e7 */
        /*00a0*/                   EXIT;                                        /* 0x8000000000001de7 */
        /*00a8*/                   BRA 0xa8;                                    /* 0x4003ffffe0001de7 */
        /*00b0*/                   NOP;                                         /* 0x4000000000001de4 */
        /*00b8*/                   NOP;                                         /* 0x4000000000001de4 */
		.............................................



Fatbin ptx code:
================
arch = sm_30
code version = [6,0]
producer = cuda
host = linux
compile_size = 64bit
compressed
```  

## use `reinterpret_cast<int4*>`, or `reinterpret_cast<float4*>` and make sure to process, deal with, elements that are "remainders"  

Improve performance of this operation by using vectorized load and store instructions `LD.E.{64,128}`, and `ST.E.{64,128}`.  
These operations also load and store data, but do so in 64- or 128-bit widths.  

Dereferencing those pointers (e.g. `reinterpret_cast<int4*>(d_in)`, `reinterpret_cast<float4*>(d_in)` will cause compiler to generate vectorized instructions.  
However, 1 *important caveat*: these instructions require aligned data.  
- Device allocated memory automatically aligned to a multiple of the size of the data type  
	* but if you offset pointer, offset must also be aligned  
	* e.g. `reinterpret_cast<int2*>(d_in+1)` is invalid because `d_in+1` not aligned to multiple of `sizeof(int2)`  
	* you can safely offset arrays if you use an "aligned" offset, e.g. `reinterpret_cast<int2*>(d_in+2)`  
	* also, can generate vectorized loads using *structures* as long as structure is a power of 2 bytes in size:  
	```  
	struct Foo (int a, int b, double c); // 16 bytes in size 
	Foo *a, *b;
	... 
	a[i]=b[i];
	```  

For `__global__ void device_copy_vector2_kernel`,  
Luitjens obtained the following from inspecting the SASS:  

```  
/*0088*/                IMAD R10.CC, R3, R5, c[0x0][0x140]              
/*0090*/                IMAD.HI.X R11, R3, R5, c[0x0][0x144]            
/*0098*/                IMAD R8.CC, R3, R5, c[0x0][0x148]             
/*00a0*/                LD.E.64 R6, [R10]                                      
/*00a8*/                IMAD.HI.X R9, R3, R5, c[0x0][0x14c]           
/*00c8*/                ST.E.64 [R8], R6
```  	

Notice compiler generates  
- `LD.E.64` and `ST.E.64`  
All other instructions are same.    
However, it's important to note that there'll be *half* as many instructions executed because loop only executes `N/2` times.  (EY: 20180104 or does he mean only half the total number of threads will be needed to do work?)  
"This 2x improvement in instruction count is very important in instruction-bound or latency bound kernels."  

I obtained for SASS  

```  
	code for sm_30
		Function : _Z26device_copy_vector2_kernelPiS_i
	.headerflags    @"EF_CUDA_SM30 EF_CUDA_PTX_SM(EF_CUDA_SM30)"
                                                                                /* 0x2242323042004307 */
        /*0008*/                   MOV R1, c[0x0][0x44];                        /* 0x2800400110005de4 */
        /*0010*/                   MOV R2, c[0x0][0x150];                       /* 0x2800400540009de4 */
        /*0018*/                   MOV32I R4, 0x2;                              /* 0x1800000008011de2 */
        /*0020*/                   S2R R0, SR_CTAID.X;                          /* 0x2c00000094001c04 */
        /*0028*/                   SSY 0xe0;                                    /* 0x60000002c0001c07 */
        /*0030*/                   S2R R3, SR_TID.X;                            /* 0x2c0000008400dc04 */
        /*0038*/                   IMAD.U32.U32.HI R2, R2, R4, c[0x0][0x150];   /* 0x2008800540209c43 */
                                                                                /* 0x22c042f2f2c28237 */
        /*0048*/                   IMAD R0, R0, c[0x0][0x28], R3;               /* 0x20064000a0001ca3 */
        /*0050*/                   SHR R6, R2, 0x1;                             /* 0x5800c00004219c23 */
        /*0058*/                   ISETP.GE.AND P0, PT, R0, R6, PT;             /* 0x1b0e00001801dc23 */
        /*0060*/               @P0 NOP.S;                                       /* 0x40000000000001f4 */
        /*0068*/                   MOV R7, R0;                                  /* 0x280000000001dde4 */
        /*0070*/                   ISCADD R2.CC, R7, c[0x0][0x140], 0x3;        /* 0x4001400500709c63 */
        /*0078*/                   MOV32I R8, 0x8;                              /* 0x1800000020021de2 */
                                                                                /* 0x23f2828202c04287 */
        /*0088*/                   IMAD.HI.X R3, R7, R8, c[0x0][0x144];         /* 0x209080051070dce3 */
        /*0090*/                   LD.E.64 R2, [R2];                            /* 0x8400000000209ca5 */
        /*0098*/                   ISCADD R4.CC, R7, c[0x0][0x148], 0x3;        /* 0x4001400520711c63 */
        /*00a0*/                   IMAD.HI.X R5, R7, R8, c[0x0][0x14c];         /* 0x2090800530715ce3 */
        /*00a8*/                   MOV R8, c[0x0][0x34];                        /* 0x28004000d0021de4 */
        /*00b0*/                   IMAD R7, R8, c[0x0][0x28], R7;               /* 0x200e4000a081dca3 */
        /*00b8*/                   ISETP.LT.AND P0, PT, R7, R6, PT;             /* 0x188e00001871dc23 */
                                                                                /* 0x2202e2c282f2e047 */
        /*00c8*/                   ST.E.64 [R4], R2;                            /* 0x9400000000409ca5 */
        /*00d0*/               @P0 BRA 0x70;                                    /* 0x4003fffe600001e7 */
        /*00d8*/                   NOP.S;                                       /* 0x4000000000001df4 */
        /*00e0*/                   IMAD R6, R6, 0x2, R0;                        /* 0x2000c00008619ca3 */
        /*00e8*/                   ISETP.GE.AND P0, PT, R6, c[0x0][0x150], PT;  /* 0x1b0e40054061dc23 */
        /*00f0*/               @P0 EXIT;                                        /* 0x80000000000001e7 */
        /*00f8*/                   ISCADD R2.CC, R6, c[0x0][0x140], 0x2;        /* 0x4001400500609c43 */
                                                                                /* 0x23f28042c04282b7 */
        /*0108*/                   MOV32I R5, 0x4;                              /* 0x1800000010015de2 */
        /*0110*/                   IMAD.HI.X R3, R6, R5, c[0x0][0x144];         /* 0x208a80051060dce3 */
        /*0118*/                   LD.E R2, [R2];                               /* 0x8400000000209c85 */
        /*0120*/                   ISCADD R4.CC, R6, c[0x0][0x148], 0x2;        /* 0x4001400520611c43 */
        /*0128*/                   IMAD.HI.X R5, R6, R5, c[0x0][0x14c];         /* 0x208a800530615ce3 */
        /*0130*/                   IADD R6, R0, R6;                             /* 0x4800000018019c03 */
        /*0138*/                   ISETP.LT.AND P0, PT, R6, c[0x0][0x150], PT;  /* 0x188e40054061dc23 */
                                                                                /* 0x2000000002f2e047 */
        /*0148*/                   ST.E [R4], R2;                               /* 0x9400000000409c85 */
        /*0150*/               @P0 BRA 0xf8;                                    /* 0x4003fffe800001e7 */
        /*0158*/                   EXIT;                                        /* 0x8000000000001de7 */
        /*0160*/                   BRA 0x160;                                   /* 0x4003ffffe0001de7 */
        /*0168*/                   NOP;                                         /* 0x4000000000001de4 */
        /*0170*/                   NOP;                                         /* 0x4000000000001de4 */
        /*0178*/                   NOP;                                         /* 0x4000000000001de4 */
		..............................................

```  
`LD.E.64` at `/*0090*/`, `ST.E.64` at `/**00c8*/`.  

EY:20180104 : Why are there so many instructions?  Is it because of `cudaMallocManaged`? 

For `__global__ void device_copy_vector4_kernel`,  
corresponding SASS obtained by Luitjens was  
```  
/*0090*/                IMAD R10.CC, R3, R13, c[0x0][0x140]              
/*0098*/                IMAD.HI.X R11, R3, R13, c[0x0][0x144]            
/*00a0*/                IMAD R8.CC, R3, R13, c[0x0][0x148]               
/*00a8*/                LD.E.128 R4, [R10]                               
/*00b0*/                IMAD.HI.X R9, R3, R13, c[0x0][0x14c]             
/*00d0*/                ST.E.128 [R8], R4
```  
see generated `LD.E.128`, and `ST.E.128`.  

I obtained  
```  
	code for sm_30
		Function : _Z26device_copy_vector4_kernelPiS_i
	.headerflags    @"EF_CUDA_SM30 EF_CUDA_PTX_SM(EF_CUDA_SM30)"
                                                                                /* 0x2202323042804307 */
        /*0008*/                   MOV R1, c[0x0][0x44];                        /* 0x2800400110005de4 */
        /*0010*/                   MOV R3, c[0x0][0x150];                       /* 0x280040054000dde4 */
        /*0018*/                   MOV32I R0, 0x4;                              /* 0x1800000010001de2 */
        /*0020*/                   SHR R3, R3, 0x1f;                            /* 0x5800c0007c30dc23 */
        /*0028*/                   SSY 0xe8;                                    /* 0x60000002e0001c07 */
        /*0030*/                   S2R R2, SR_CTAID.X;                          /* 0x2c00000094009c04 */
        /*0038*/                   S2R R4, SR_TID.X;                            /* 0x2c00000084011c04 */
                                                                                /* 0x2202f2f2c2820277 */
        /*0048*/                   IMAD.U32.U32.HI R3, R3, R0, c[0x0][0x150];   /* 0x200080054030dc43 */
        /*0050*/                   IMAD R2, R2, c[0x0][0x28], R4;               /* 0x20084000a0209ca3 */
        /*0058*/                   SHR R3, R3, 0x2;                             /* 0x5800c0000830dc23 */
        /*0060*/                   ISETP.GE.AND P0, PT, R2, R3, PT;             /* 0x1b0e00000c21dc23 */
        /*0068*/               @P0 NOP.S;                                       /* 0x40000000000001f4 */
        /*0070*/                   MOV R10, R2;                                 /* 0x2800000008029de4 */
        /*0078*/                   ISCADD R4.CC, R10, c[0x0][0x140], 0x4;       /* 0x4001400500a11c83 */
                                                                                /* 0x228232c04282b047 */
        /*0088*/                   MOV32I R9, 0x10;                             /* 0x1800000040025de2 */
        /*0090*/                   MOV R11, c[0x0][0x34];                       /* 0x28004000d002dde4 */
        /*0098*/                   IMAD.HI.X R5, R10, R9, c[0x0][0x144];        /* 0x2092800510a15ce3 */
        /*00a0*/                   LD.E.128 R4, [R4];                           /* 0x8400000000411cc5 */
        /*00a8*/                   ISCADD R8.CC, R10, c[0x0][0x148], 0x4;       /* 0x4001400520a21c83 */
        /*00b0*/                   IMAD.HI.X R9, R10, R9, c[0x0][0x14c];        /* 0x2092800530a25ce3 */
        /*00b8*/                   IMAD R10, R11, c[0x0][0x28], R10;            /* 0x20144000a0b29ca3 */
                                                                                /* 0x22e2c282f2e043f7 */
        /*00c8*/                   ISETP.LT.AND P0, PT, R10, R3, PT;            /* 0x188e00000ca1dc23 */
        /*00d0*/                   ST.E.128 [R8], R4;                           /* 0x9400000000811cc5 */
        /*00d8*/               @P0 BRA 0x78;                                    /* 0x4003fffe600001e7 */
        /*00e0*/                   NOP.S;                                       /* 0x4000000000001df4 */
        /*00e8*/                   IMAD R3, R3, 0x4, R2;                        /* 0x2004c0001030dca3 */
        /*00f0*/                   ISETP.GE.AND P0, PT, R3, c[0x0][0x150], PT;  /* 0x1b0e40054031dc23 */
        /*00f8*/               @P0 EXIT;                                        /* 0x80000000000001e7 */
                                                                                /* 0x23f28202c04282c7 */
        /*0108*/                   ISCADD R4.CC, R3, c[0x0][0x140], 0x2;        /* 0x4001400500311c43 */
        /*0110*/                   IMAD.HI.X R5, R3, R0, c[0x0][0x144];         /* 0x2080800510315ce3 */
        /*0118*/                   LD.E R5, [R4];                               /* 0x8400000000415c85 */
        /*0120*/                   ISCADD R6.CC, R3, c[0x0][0x148], 0x2;        /* 0x4001400520319c43 */
        /*0128*/                   IMAD.HI.X R7, R3, R0, c[0x0][0x14c];         /* 0x208080053031dce3 */
        /*0130*/                   IADD R3, R2, R3;                             /* 0x480000000c20dc03 */
        /*0138*/                   ISETP.LT.AND P0, PT, R3, c[0x0][0x150], PT;  /* 0x188e40054031dc23 */
                                                                                /* 0x2000000002f2e047 */
        /*0148*/                   ST.E [R6], R5;                               /* 0x9400000000615c85 */
        /*0150*/               @P0 BRA 0x100;                                   /* 0x4003fffea00001e7 */
        /*0158*/                   EXIT;                                        /* 0x8000000000001de7 */
        /*0160*/                   BRA 0x160;                                   /* 0x4003ffffe0001de7 */
        /*0168*/                   NOP;                                         /* 0x4000000000001de4 */
        /*0170*/                   NOP;                                         /* 0x4000000000001de4 */
        /*0178*/                   NOP;                                         /* 0x4000000000001de4 */
		..............................................
```  

### Correct way to loop over all the elements in an array of length L, even when L is not a power of 2 or 4, for vector loading    

e.g. `../cooperative_groups/cg_eg3.cu`, `__device__ int thread_sum(int *, int L)`  

```  
	unsigned int k_x = threadIdx.x + blockDim.x*blockIdx.x; 
	
	/* increment by blockDim.x*gridDim.x, so that a single thread will do all the 
	 * "work" needed done on n, especially if n >= gridDim.x*blockDim.x = N_x*M_x */	
	for (int i=k_x; 
			i < L/4; 
			i += blockDim.x * gridDim.x) 
	{
		int4 in = ((int4*) input)[i]; 
		sum += in.x + in.y + in.z + in.w; 
	}
	// process remaining elements
	for (unsigned int idx= k_x + L/4*4; idx < L; idx += 4 ) {
		sum += input[idx];
	}
```  

and launch these number of (thread) blocks, given `M_x` number of threads in a single (thread) block (in x-direction):  

```  
	// notice how we're only launching 1/4 of L threads
	N_x = min( MAX_BLOCKS, ((L/4 + M_x - 1)/ M_x)); 

sum_kernel<<<N_x,M_x,>>>(sum.get(),input.get(),L); 

```

Note the 2 loops:  
```  
	unsigned int k_x = threadIdx.x + blockDim.x*blockIdx.x; 
	
	/* increment by blockDim.x*gridDim.x, so that a single thread will do all the 
	 * "work" needed done on n, especially if n >= gridDim.x*blockDim.x = N_x*M_x */	
	for (int i=k_x; 
			i < L/4; 
			i += blockDim.x * gridDim.x) 
```  
and  

```  
	// process remaining elements
	for (unsigned int idx= k_x + L/4*4; idx < L; idx += 4 ) {
```  




### Short note on number of (thread) blocks to launch, N_x (in the x-direction); how to get device GPU's max. blocks and compare to how many blocks you need  

Get the `.maxGridsize` of a single device GPU inside of `main` function or as a function:  
```  
cudaDeviceProp prop;
int count;
cudaGetDeviceCount(&count); 
int MAXGRIDSIZE;
if (count >0) {
	cudaGetDeviceProperties( &prop, 0); 
	MAXGRIDSIZE = prop.maxGridSize[0];
} else { return EXIT_FAILURE; }
```  
or as a function (call) `get_maxGridSize()`  
```  
size_t get_maxGridSize() {
	cudaDeviceProp prop;
	int count;
	cudaGetDeviceCount(&count);
	size_t MAXGRIDSIZE; 
	if (count>0) {
		cudaGetDeviceProperties(&prop, 0);
		MAXGRIDSIZE = prop.maxGridSize[0]; 
		return MAXGRIDSIZE; 
	} else { return EXIT_FAILURE; }
}; 
``` 
On my nVidia GeForce GTX 980Ti (I need a hardware donation; please donate a TITAN V or GTX 1080Ti if you find my work useful!), `.maxGridSize[0]` (in x-direction) is   
```  
2147483647
```  
which is 2^31, which should be the theoretical max. value that can be stored in a 32-bit unsigned int.  What if you're doing multi-GPUs, with more than 2^(31) threads available?  Then I would think you'd need `size_t` to store this value, not `unsigned int`.  

Then do this formula:  
```  
N_x = (L_x + M_x - 1)/M_x; 
```  
where `M_x` is the number of threads in a single (thread) block, and `L` is either 2 possibilities:  
- the total size of the array you have, `L`, or     
- max. number of threads allowed on the device GPU, that you found from, say, `get_maxGridSize()`, `MAXGRIDSIZE`.  
and then you calculate `N_x_needed`, or `N_x_MAX`, respectively.  

Clearly, you want to take the `min` of `N_x_needed` and `N_x_MAX` (because otherwise, you'd launch more threads than physically allowed on device GPU hardware).  
However, it is a question of whether you want to do this check in the `main` driver function, or inside a driver function.  If the latter, do this, for example:  
```  
void device_copy_vector4(int* d_in, int* d_out, int N, unsigned int MAX_BLOCKS) {
	int threads = 128; 
	int blocks=min((N/4+threads-1)/ threads, MAX_BLOCKS); 
	
	device_copy_vector4_kernel<<<blocks,threads>>>(d_in,d_out,N); 
}
```  



