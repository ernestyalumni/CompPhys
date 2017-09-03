/**
 * @name : Ex01_singleprocess.cu
 * @ref  : http://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/index.html#singleprothrdmultidev  
 * @note : COMPILATION TIPS:
 *			nvcc -std=c++11 -I ~/nccl/include -L ~/nccl/lib -lnccl Ex01_singleprocess.cu -o Ex01_singleprocess.ex 
 * */
 /* 
 * COMPILATION TIP
 * nvcc -std=c++11 -I ~/nccl/include -L ~/nccl/lib -lnccl nccl_playground.cu -o nccl_playground.exe
 * 
 * You may receive the following errors:
 * tmpxft_0000761b_00000000-4_nccl_playground.cudafe1.cpp:(.text+0x40f): undefined reference to `ncclCommInitAll'
collect2: error: ld returned 1 exit status
 * 
 * That's because we need to include the library files, those *.so files some how in the system.  
 * If you don't want to do a permanent setup that'll require root access (and then making the hard symbolic link and all), 
 * follow this page http://www.cplusplus.com/forum/unices/33532/
 * and do the 2 flags -L ~/nccl/lib, i.e. -L /path/to/my/library/folder and 
 * -lnccl i.e. -l<libname>
 * 
 * */
#include <stdio.h>
#include "cuda_runtime.h"
#include "nccl.h"

#include <vector> // std::vector

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

int main(int argc, char* argv[])
{
	/*
  ncclComm_t comms[4];

  //managing 4 devices	  
  int nDev = 4;
  int size = 32*1024*1024;
  int devs[4] = { 0, 1, 2, 3 };
	*/

  // managing 1 device
  ncclComm_t comms[1];
  int nDev = 1;
  int size = 32*1024*1024;
  printf(" size : %d \n ", size);
  int devs[1] = { 0 };
  

  //allocating and initializing device buffers
  float** sendbuff = (float**)malloc(nDev * sizeof(float*));
  float** recvbuff = (float**)malloc(nDev * sizeof(float*));
  cudaStream_t* s = (cudaStream_t*)malloc(sizeof(cudaStream_t)*nDev);

  for (int i = 0; i < nDev; ++i) {
	printf(" \n i : %d \n ", i );
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaMalloc(sendbuff + i, size * sizeof(float)));
    CUDACHECK(cudaMalloc(recvbuff + i, size * sizeof(float)));
    CUDACHECK(cudaMemset(sendbuff[i], 1.f, size * sizeof(float)));
    CUDACHECK(cudaMemset(recvbuff[i], 0.f, size * sizeof(float)));
    CUDACHECK(cudaStreamCreate(s+i));
  }

	cudaDeviceSynchronize();

	// read the input 
	std::vector<float> f_vec(size,1.f);
	cudaMemcpy(sendbuff[0], f_vec.data(), size*sizeof(float), cudaMemcpyHostToDevice);
	printf(" f_vec[0] : %f , f_vec[1] : %f , f_vec[size-2] : %f , f_vec[size-1] : %f \n ", 
		f_vec[0], f_vec[1], f_vec[size-2], f_vec[size-1] );	


  //initializing NCCL
  NCCLCHECK(ncclCommInitAll(comms, nDev, devs));

  //calling NCCL communication API. Group API is required when using
  //multiple devices per thread
	NCCLCHECK(ncclGroupStart());
	for (int i = 0; i < nDev; ++i) {
		printf(" \n i : %d \n ", i );
		NCCLCHECK(ncclAllReduce((const void*)sendbuff[i], (void*)recvbuff[i], size, ncclFloat, ncclSum,
									comms[i], s[i])); 
	}
	NCCLCHECK(ncclGroupEnd());

  //synchronizing on CUDA streams to wait for completion of NCCL operation
  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaStreamSynchronize(s[i]));
  }

	// read the output 
	std::vector<float> g_vec(size,0.f);
	
	cudaMemcpy(g_vec.data(), recvbuff[0], size*sizeof(float), cudaMemcpyDeviceToHost);
	printf(" g_vec[0] : %f , g_vec[1] : %f , g_vec[size-2] : %f , g_vec[size-1] : %f \n ", 
		g_vec[0], g_vec[1], g_vec[size-2], g_vec[size-1] );	



  //free device buffers
  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaFree(sendbuff[i]));
    CUDACHECK(cudaFree(recvbuff[i]));
  }

  //finalizing NCCL
  for(int i = 0; i < nDev; ++i)
      ncclCommDestroy(comms[i]);

  printf("Success \n");
  return 0;
}

