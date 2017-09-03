/**
 * @file   : nccl_playground.cu
 * @brief  : Playground for using nccl
 * @author : Ernest Yeung <ernestyalumni@gmail.com>
 * @date   : 20170901  
 * @ref    :  
 * 
 * If you find this code useful, feel free to donate directly and easily at this direct PayPal link: 
 * 
 * https://www.paypal.com/cgi-bin/webscr?cmd=_donations&business=ernestsaveschristmas%2bpaypal%40gmail%2ecom&lc=US&item_name=ernestyalumni&currency_code=USD&bn=PP%2dDonationsBF%3abtn_donateCC_LG%2egif%3aNonHosted 
 * 
 * which won't go through a 3rd. party such as indiegogo, kickstarter, patreon.  
 * Otherwise, I receive emails and messages on how all my (free) material on 
 * physics, math, and engineering have helped students with their studies, 
 * and I know what it's like to not have money as a student, but love physics 
 * (or math, sciences, etc.), so I am committed to keeping all my material 
 * open-source and free, whether or not 
 * sufficiently crowdfunded, under the open-source MIT license: 
 * 	feel free to copy, edit, paste, make your own versions, share, use as you wish.  
 *  Just don't be an asshole and not give credit where credit is due.  
 * Peace out, never give up! -EY
 * 
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
#include <iostream> // std::cout
#include <memory> 	// std::unique_ptr
#include <vector> 	// std::vector

#include "cuda_runtime.h"

#include "nccl.h"

int main(int argc, char* argv[]) {
	int nvis = 0;
	cudaGetDeviceCount(&nvis);
	if (nvis == 0) {
		std::cout << "No GPUs found " << std::endl; 
		exit(EXIT_FAILURE);
	}
	std::cout << " nvis : " << nvis << std::endl;

	/**
	 * cf. https://devtalk.nvidia.com/default/topic/414231/cuda-programming-and-performance/getting-device-39-s-free-mem/
	 * cf. https://stackoverflow.com/questions/7386990/cuda-identifier-cudamemgetinfo-is-undefined
	 * */
	size_t d_N_free_mem, d_N_total_mem; 
	cudaMemGetInfo(&d_N_free_mem, &d_N_total_mem);
	std::cout << " free = " << d_N_free_mem << " total = " << d_N_total_mem << std::endl; 

	constexpr const float percent_of_free_mem = 0.75f;
	
	auto N_floats_free = ((size_t) d_N_free_mem * percent_of_free_mem / sizeof(float) );

	std::cout << " N_floats_free = number of floats that can be allocated at " << percent_of_free_mem << 
		" of free GPU memory : " << N_floats_free << std::endl;

//	constexpr const int min_size = ;
	constexpr const int n_max = 42;
	size_t max_size = n_max * sizeof(float);
	

	
	//float* d_out;
//	std::unique_ptr<float[]> d_out( new float[n_max], []( float *ptr){ cudaFree(ptr); } );
//	std::unique_ptr<float[], cudaError_t*> d_out( new float[n_max], [](float *ptr ){cudaFree(ptr);});  // works
//	std::unique_ptr<float[], void *> d_out( new float[n_max], [](float *ptr ){cudaFree(ptr);});  // error no instance 
//	std::unique_ptr<float[]> d_out( new float[n_max] );  // works
//	std::unique_ptr<float[], decltype( [](float *ptr){ cudaFree(ptr); }*> 
//		d_out( new float[n_max],[](float *ptr){ cudaFree(ptr); }* ); 
//	std::shared_ptr<float> d_out( new float[n_max], [](float *ptr){ cudaFree(ptr); } );

//	cudaMalloc((void **) &d_out, max_size); // works
//	cudaMalloc((void **) &d_out, max_size);

//	cudaMalloc((void **) &d_out.get(), max_size);
//	cudaMalloc(&&d_out.get(), max_size); // error: expression must have class type
	
//	cudaFree(&d_out);  // works
	
	/*  
	 * This works, but defeats the purpose of using a smart pointer:
	 * 
	 * std::unique_ptr<float[]> d_out( new float[n_max] );  // works
	 * cudaMalloc((void **) &d_out, max_size); // works
	 * cudaFree(&d_out);  // works
	 */


	char busid[32] = {0};
	cudaDeviceGetPCIBusId(busid, 32, 0);
	std::cout <<"# Rank " << 0 << " using device " << 0 << " [" << busid << "] " << std::endl; 


	 
	/*
	 * This also works as well: 
	 * cf. https://stackoverflow.com/questions/10319009/unique-ptrt-lambda-custom-deleter-for-array-specialization
	 * */ 
/*	auto deleter=[&](float* ptr){ cudaFree(ptr); };
	std::unique_ptr<float[], decltype(deleter)> d_out(new float[n_max], deleter);
	cudaMalloc((void **) &d_out, max_size);
*/
	const int ARRAY_SIZE { 1<<12 } ;
	std::cout << " ARRAY_SIZE { 1 << 12 } : " << ARRAY_SIZE << std::endl; 

	// generate input array on host
	std::vector<float> f_vec;
	for (int i = 0; i < ARRAY_SIZE; ++i) {
		f_vec.push_back(i+1) ; }
	
	auto deleter=[&](float* ptr){ cudaFree(ptr); };
	std::unique_ptr<float[], decltype(deleter)> d_in(new float[ARRAY_SIZE], deleter);
	cudaMalloc((void **) &d_in, ARRAY_SIZE * sizeof(float));

	std::unique_ptr<float[], decltype(deleter)> d_out(new float[ARRAY_SIZE], deleter);
	cudaMalloc((void **) &d_out, ARRAY_SIZE * sizeof(float));

	cudaMemcpy(d_in.get(), f_vec.data(), sizeof(float) * ARRAY_SIZE, cudaMemcpyHostToDevice);

	// We can also make smart pointers for cudaStream, i.e. CUDA stream, if we had multiple devices to deal with
//	auto stream_deleter=[&](cudaStream_t stream){ cudaStreamDestroy( &stream_ptr ); };
//	std::unique_ptr<cudaStream_t, decltype(stream_deleter)> stream(new cudaStream_t,stream_deleter);
	
	// we can either make a stream for 1 device with a raw pointer:
	/*
	cudaStream_t stream;
	cudaStreamCreate(&stream);
	*/ 
	// or this way:
	auto stream_deleter=[&](cudaStream_t* stream){ cudaStreamDestroy( *stream ); };
	std::unique_ptr<cudaStream_t, decltype(stream_deleter)> stream(new cudaStream_t, stream_deleter);
	
	
	
	ncclComm_t comms[1];
	
	// managing 1 device
	int devs[1] = { 0 };
	
	// initializing NCCL
	ncclCommInitAll(comms,nvis,devs);  // nvis or nDev or number of devices should be 1 if there's only 1 GPU
	

	cudaSetDevice(0);

	ncclAllReduce(d_in.get(), d_out.get(), ARRAY_SIZE,ncclFloat,ncclSum,comms[0],*stream.get());
	
	// read the output
	// generate output array on host
	std::vector<float> g_vec(ARRAY_SIZE,0.f);
	cudaMemcpy(g_vec.data(), d_out.get(),sizeof(float) * ARRAY_SIZE, cudaMemcpyDeviceToHost);
	std::cout << " g_vec[0] : " << g_vec[0] << " g_vec[1] : " << g_vec[1] << 
		" g_vec[ARRAY_SIZE-2] : " << g_vec[ARRAY_SIZE-2] << " g_vec[ARRAY_SIZE-1] : " << g_vec[ARRAY_SIZE-1] << std::endl;

	
	// this is the way to destroy a stream at end of its life
//	cudaStreamDestroy(stream);

	exit(EXIT_SUCCESS);
}
