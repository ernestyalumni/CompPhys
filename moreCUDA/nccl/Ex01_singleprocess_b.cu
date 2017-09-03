/**
 * @file   : Ex01_singleprocess_b.cu
 * @brief  : Single Process, Single Device
 * @author : Ernest Yeung <ernestyalumni@gmail.com>
 * @date   : 20170902  
 * @ref    : http://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/index.html#singleprothrdmultidev 
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

#include "nccl.h"

int main(int argc, char* argv[])
{
	// managing 1 device
	auto comm_deleter=[&](ncclComm_t* comm){ ncclCommDestroy( *comm ); };
	std::unique_ptr<ncclComm_t, decltype(comm_deleter)> comm(new ncclComm_t, comm_deleter);	


	// sanity check of number of GPUs 
	int nDev = 0;
	cudaGetDeviceCount(&nDev);
	if (nDev == 0) {
		std::cout << "No GPUs found " << std::endl; 
		exit(EXIT_FAILURE);
	}
	std::cout << " nDev (number of devices) : " << nDev << std::endl;
	// END of sanity check of number of GPUs 


	int size = 32 * 1024 * 1024;
	std::cout << " size : " << size << std::endl; 
	int devs[1] = {0};
	
	// generate input vector/array on host
	std::vector<float> f_vec(size,2.f);
	

	// device pointers
	auto deleter=[&](float* ptr){ cudaFree(ptr); };
	std::unique_ptr<float[], decltype(deleter)> d_in(new float[size], deleter);
	cudaMalloc((void **) &d_in, size * sizeof(float));

	std::unique_ptr<float[], decltype(deleter)> d_out(new float[size], deleter);
	cudaMalloc((void **) &d_out, size * sizeof(float));


	// CUDA stream smart pointer stream
	auto stream_deleter=[&](cudaStream_t* stream){ cudaStreamDestroy( *stream ); };
	std::unique_ptr<cudaStream_t, decltype(stream_deleter)> stream(new cudaStream_t, stream_deleter);
	cudaStreamCreate(stream.get());
	

	cudaMemcpy(d_in.get(), f_vec.data(), size*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemset(d_out.get(), 0.f, size*sizeof(float));

	cudaMemcpy(f_vec.data(), d_out.get(), size*sizeof(float), cudaMemcpyDeviceToHost);


	cudaDeviceSynchronize();

	//initializing NCCL
	ncclCommInitAll(comm.get(), nDev, devs);

	// number of ranks in a communicator
	int count =0;
	ncclCommCount(*comm.get(),&count);
	std::cout << " number of ranks in a communicator, using ncclCommCount : " << count << std::endl;

	ncclAllReduce( d_in.get(), d_out.get(), size, ncclFloat, ncclSum, *comm.get(), *stream.get() );


	// read out output in host
	cudaMemcpy(f_vec.data(), d_out.get(), size*sizeof(float), cudaMemcpyDeviceToHost);
	std::cout << "On the device:  f_vec[0] : " << f_vec[0] << ", f_vec[1] : " << f_vec[1] << 
		", f_vec[2] : " << f_vec[2] << ", f_vec[3] : " << f_vec[3] << 
		", f_vec.back() : " << f_vec.back() << std::endl;
	for (int idx=4; idx < 32+4 ; idx++) {
		std::cout << idx << " : " << f_vec[idx] << " " ; 
	}


	cudaDeviceReset();
		
	return 0;
}
