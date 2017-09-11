/**
 * @file   : commonmultiplestreams.cu
 * @brief  : Common pattern for dispatching CUDA operations to multiple streams 
 * @author : Ernest Yeung <ernestyalumni@gmail.com>
 * @date   : 20170904  
 * @ref    : http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-concurrent-execution
 * 		   : 3.2.5. Asynchronous Concurrent Execution of CUDA Toolkit v8.0, 3. Programming Interface  
 * 		   : John Cheng, Max Grossman, Ty McKercher. Professional CUDA C Programming. 1st Ed. Wrox. 2014
 * 		   : Ch. 6 Streams and Concurrency; pp. 271 
 * 		   : Shane Cook, CUDA Programming, pp. 278 Multi-CPU and Multi-GPU Solutions   
 * If you find this code useful, feel free to donate directly and easily at this direct PayPal link: 
 * 
 * @note : if you have problems with Segmentation Fault, I obtained Segmentation Fault when 
 * I entered and exited Sleep Mode on Fedora 23 Workstation linux.  When 
 * I turned off and turned back on the computer, it worked.  

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
 * nvcc -std=c++11 smart_ptrs_arith.cu -o smart_ptrs_arith.exe
 * 
 * @note : if you have problems with Segmentation Fault, I obtained Segmentation Fault when 
 * I entered and exited Sleep Mode on Fedora 23 Workstation linux.  When 
 * I turned off and turned back on the computer, it worked.  
 * */
#include <iostream>

/** 
 * @fn    : evenoddincrement
 * @brief : increment different values for even indexed entries, 
 * 			increment different values for odd indexed entries
 */
__global__ void evenoddincrement(float *g_data, int even_inc, int odd_inc) {
	int tx = threadIdx.x + blockIdx.x * blockDim.x; 
	if ((tx % 2) == 0) {
		g_data[tx] += even_inc; 
	}
	else {
		g_data[tx] += odd_inc;
	}
}



int main(int argc, char *argv[]) {

	int Nstreams = (1<<1) ;	// (total) number of streams, 1<<1 = 2 
	std::cout << " Nstreams (number of streams) : " << Nstreams << std::endl; 

	int N = (1<<4)*(1<<10)*(1<<10); // 16 * 1024*1024
	std::cout << " N : " << N << std::endl;
	int Nbytes = N * sizeof(float);
	int N_i = N/Nstreams; // number of elements for a single stream
	
	// allocating and initializing host array of arrays
	float** h_ptrs; 
	float** h_outputs;
	cudaMallocHost((float**)&h_ptrs, Nbytes);
	cudaMallocHost((float**)&h_outputs, Nbytes);
	for (int idx_stream=0; idx_stream < Nstreams; ++idx_stream) {
		cudaMallocHost( h_ptrs + idx_stream, N_i * sizeof(float)); 
		cudaMallocHost( h_outputs + idx_stream, N_i * sizeof(float)); 
	}

	// actual initialization of each of the host arrays on pinned host memory
	for (int idx_stream=0; idx_stream < Nstreams; ++idx_stream) {
		for (int idx=0; idx<N_i; ++idx) {
			int i = idx + idx_stream * N_i; 
			(h_ptrs[idx_stream])[idx] = ((float) i ); 
			(h_outputs[idx_stream])[idx] = 0.f;
		}
	}
	
	// allocating and initializing device array of arrays
	// float** d_ptrs = (float**)malloc( Nstreams * sizeof(float*)); 
	float** d_ptrs;
	cudaMallocHost((float**)&d_ptrs, Nstreams*sizeof(float*));
	for (int idx_stream=0; idx_stream < Nstreams; ++idx_stream) {
		cudaMalloc(d_ptrs + idx_stream, N_i * sizeof(float));
		cudaMemset(d_ptrs[idx_stream], 0.f, N_i * sizeof(float));
	}
	
	// allocate and initialize array of streams
	cudaStream_t* streams = (cudaStream_t*)malloc(sizeof(cudaStream_t)*Nstreams);
	for (int idx_stream=0; idx_stream < Nstreams; ++idx_stream) {
		cudaStreamCreate(streams + idx_stream);
	}

	/* *** grid block dimensions ** */
	dim3 blockDims ( (1<<5), 1) ; // 1<<5 = 32 
	dim3 gridDims ( (N_i + blockDims.x - 1)/ blockDims.x, 1) ; 
	/* END of grid block dimensions */

	
	// dispatch CUDA operations to multiple streams
	for (int idx_stream =0; idx_stream < Nstreams; idx_stream++ ) { 
		cudaMemcpyAsync(d_ptrs[idx_stream], h_ptrs[idx_stream], N_i * sizeof(float), 
			cudaMemcpyHostToDevice, streams[idx_stream]);
		evenoddincrement<<<gridDims, blockDims>>>(d_ptrs[idx_stream], 2,3); 
		cudaMemcpyAsync(h_outputs[idx_stream], d_ptrs[idx_stream], N_i * sizeof(float), 
			cudaMemcpyDeviceToHost, streams[idx_stream]);
		
	}
	
	for (int idx_stream =0; idx_stream < Nstreams; idx_stream++ ) {
		cudaStreamSynchronize(streams[idx_stream]); 
	}

	// sanity check: print out results  
	const int DISPLAYSIZE = (1<<5); // specify how many digits to display on the screen
	for (int i=0; i < DISPLAYSIZE; i++) {
		std::cout << " i : " << i << ", (h_outputs[0])[i] : " << (h_outputs[0])[i] << "; "; }
	std::cout << std::endl;
		
	
	

	// release streams
	for (int idx_stream = 0; idx_stream < Nstreams; ++idx_stream) {
		cudaStreamDestroy(streams[idx_stream]);
	}

	//free device buffers
	for (int idx_stream = 0; idx_stream < Nstreams; ++idx_stream) {
		cudaFree(d_ptrs[idx_stream]);
		cudaFreeHost( h_ptrs[idx_stream]) ;
		cudaFreeHost( h_outputs[idx_stream]) ;
		

	}

	
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaDeviceReset();	
}
