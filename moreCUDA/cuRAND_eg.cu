/** \file cuRAND_eg.cu
 * \author Ernest Yeung
 * \email  ernestyalumni@gmail.com
 * Demonstrates cuRAND use
 * cf. http://docs.nvidia.com/cuda/curand/device-api-overview.html#device-api-example
 * 
 * If you find this code useful, feel free to donate directly and easily at this direct PayPal link: 
 * 
 * https://www.paypal.com/cgi-bin/webscr?cmd=_donations&business=ernestsaveschristmas%2bpaypal%40gmail%2ecom&lc=US&item_name=ernestyalumni&currency_code=USD&bn=PP%2dDonationsBF%3abtn_donateCC_LG%2egif%3aNonHosted 
 * 
 * which won't go through a 3rd. party such as indiegogo, kickstarter, patreon.  
 * Otherwise, I receive emails and messages on how all my (free) material on physics, math, and engineering have 
 * helped students with their studies, and I know what it's like to not have money as a student, but love physics 
 * (or math, sciences, etc.), so I am committed to keeping all my material open-source and free, whether or not 
 * sufficiently crowdfunded, under the open-source MIT license: 
 * 	feel free to copy, edit, paste, make your own versions, share, use as you wish.    
 * Peace out, never give up! -EY
 * 
 * */
/**
 * Compilation tips
 * 
 *  ** EY : 20170303
 * But here's what I did, on a GeForce GTX 980 Ti
 * I wanted to include the cub library, which is in a separate folder (I downloaded, unzipped)
 * but it's not symbolically linked from root (I REALLY don't want to mess with root directory right now).
 * so I put the folder (straight downloaded from the internet) into a desired, arbitrary location;
 * in this case, I put it in `./` so that it's `./cub/`
 * Then I used the include flag -I
 * in the following manner:
 * nvcc -std=c++11 -lcurand -I./CUB/cub/ cuRAND_eg.cu -o cuRAND_eg.exe
 * 
 * Also note that I was on a GeForce GTX 980 Ti and CUDA Toolkit 8.0 with latest drivers, and so 
 * for my case, Compute or SM requirements was (very much) met
 * 
 * Compiling notes: -lcurand needed for #include <curand.h>, otherwise you get these kinds of errors:
 * tmpxft_00000d2f_00000000-4_cuRAND_eg.cudafe1.cpp:(.text+0x103): undefined reference to `curandCreateGenerator'
 * tmpxft_00000d2f_00000000-4_cuRAND_eg.cudafe1.cpp:(.text+0x114): undefined reference to `curandSetPseudoRandomGeneratorSeed'
 * tmpxft_00000d2f_00000000-4_cuRAND_eg.cudafe1.cpp:(.text+0x12b): undefined reference to `curandGenerateUniform'
 * Also, be careful of the order of compiling libraries in the future; I forgot where, but I read somewhere that order matters for which library
 * 
 **********************************************************************/
#include <vector> // std::vector
#include <iostream> // std::cout

#include <cub/util_allocator.cuh> // cub::CachingDeviceAllocator, CubDebugExit 

#include <curand.h> // curandGenerator_t, curandCreateGenerator 

//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------
cub::CachingDeviceAllocator dev_allocator(true);  // Caching allocator for device memory

//----------------------------------------------------------------------
// ANCILLARY C++ functions
//----------------------------------------------------------------------
template <typename ForwardIter>
void Print2(ForwardIter first, ForwardIter last, const char* status)
{
	std::cout << status << std::endl;
	while ( first != last)
		std::cout << *first++ << " ";
	std::cout << std::endl;
	
};


int main(int argc, char *argv[])
{
//	curandState *devStates;  // EY : 20170303 not needed if we want HOST API and only basic functionality from host?
	size_t n = 100;

	
    // Allocate problem device arrays
    float 	*dev_dat = nullptr;  // NULL, otherwise for nullptr, you're going to need the -std=c++11 flag for compilation
	unsigned int *dev_i_dat = nullptr; 

	/* Allocate n floats on device */
    CubDebugExit(dev_allocator.DeviceAllocate((void**)&dev_dat, sizeof(float) * n ));
    CubDebugExit(dev_allocator.DeviceAllocate((void**)&dev_i_dat, sizeof(unsigned int) * n ));

	
	/* Create pseudo-random number generator */
	curandGenerator_t gen;

	curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT);

	/* Set seed */
	curandSetPseudoRandomGeneratorSeed(gen,1234ULL);
	
	/* Generate n floats on device */
	curandGenerateUniform(gen, dev_dat, n);
	
	////////////////////////////////////////////////////////////////////
	// OUTPUT (BOILERPLATE) 
	////////////////////////////////////////////////////////////////////
	
	// generate output array on host
	std::vector<float> f_vec(n,0.f);
//	std::vector<float> f_vec;
	std::vector<unsigned int> i_vec(n,0);
	
	/* Copy device memory to host */
	float *hostData;	
	hostData = (float *)calloc(n,sizeof(float));	
		
	CubDebugExit(cudaMemcpy(hostData, dev_dat, sizeof(float) * n, cudaMemcpyDeviceToHost));
	CubDebugExit(cudaMemcpy(f_vec.data(), dev_dat, sizeof(float) * n, cudaMemcpyDeviceToHost));


	Print2(f_vec.begin(), f_vec.end(), "result of curandGenerateUniform");

//	std::cout << f_vec[0] << std::endl;
	for (int i=0;i<n;i++) { std::cout << hostData[i] << " "; }

	// Normal

	/* Set seed (again) */
	curandSetPseudoRandomGeneratorSeed(gen,1235ULL);

	/* Generate n floats on device */
	curandGenerateNormal(gen, dev_dat, n, 0.f, 1.f); // mean, then stddev

	/* Copy device memory to host */
	CubDebugExit(cudaMemcpy(f_vec.data(), dev_dat, sizeof(float) * n, cudaMemcpyDeviceToHost));
	Print2(f_vec.begin(), f_vec.end(), "result of curandGenerateNormal");

	// Log Normal

	/* Generate n floats on device */
	curandGenerateLogNormal(gen, dev_dat, n, 0.f, 1.f); // mean, then stddev

	/* Copy device memory to host */
	CubDebugExit(cudaMemcpy(f_vec.data(), dev_dat, sizeof(float) * n, cudaMemcpyDeviceToHost));
	Print2(f_vec.begin(), f_vec.end(), "result of curandGenerateLogNormal");

	// Poisson

	/* Generate n floats on device */
	curandGeneratePoisson(gen, dev_i_dat, n, 1.); // lambda

	/* Copy device memory to host */
	CubDebugExit(cudaMemcpy(i_vec.data(), dev_i_dat, sizeof(unsigned int) * n, cudaMemcpyDeviceToHost));
	Print2(i_vec.begin(), i_vec.end(), "result of curandGeneratePoisson");



	/* Clean up */
    if (dev_dat) CubDebugExit(dev_allocator.DeviceFree(dev_dat));

	free(hostData);

	return EXIT_SUCCESS; 
}


