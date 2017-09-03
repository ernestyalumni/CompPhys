/**
 * @file   : cudaMemset_sanitycheck.cu
 * @brief  : cudaMemset sanity check.
 * 			 This empirically demonstrates that cudaMemcpy can only set values to 0.  
 * 			 It is unclear to me still why this is the case, but it may be because
 * 			 it, be inherent design, can only set bytes to a specific value, and 
 * 			 not for specific type
 * @author : Ernest Yeung <ernestyalumni@gmail.com>
 * @date   : 20170902  
 * @ref    : https://stackoverflow.com/questions/13387101/cudamemset-does-it-set-bytes-or-integers
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

#include <iostream>  

int main(int argc, char* argv[])
{
	// count how many GPU devices
	int nDev = 0;
	cudaGetDeviceCount(&nDev);
	if (nDev == 0) {
		std::cout << "No GPUs found " << std::endl; 
		exit(EXIT_FAILURE);
	}
	std::cout << " nDev (number of devices) : " << nDev << std::endl;
	// END of counting of how many GPU devices

	constexpr const int N { 1 << 5 } ;
	std::cout << " N { 1 << 5 } : " << N << std::endl; 
	float* d_f;
	cudaMalloc(&d_f, N * sizeof(float));
	cudaMemset(d_f, 0.f , N*sizeof(float));
	
	cudaDeviceSynchronize();
	
	float* f = new float[N];
	cudaMemcpy(f,d_f,N*sizeof(float),cudaMemcpyDeviceToHost);
	
	for (int i=0; i < N; i++) {
		std::cout << i << " : " << f[i] << ", "; 
	}

	/* fills the first count (N*sizeof(float)) bytes of the memory area 
	 * pointed to by devPtr (d_f), with constant byte value value (0.f) 
	 * */
	cudaMemset(d_f, 0x15 , N*sizeof(float));
	cudaDeviceSynchronize();

	cudaMemcpy(f,d_f,N*sizeof(float),cudaMemcpyDeviceToHost);
	for (int i=0; i < N; i++) {
		std::cout << i << " : " << f[i] << ", "; 
	}

	cudaMemset(d_f, 3.f , N*sizeof(float));
	cudaDeviceSynchronize();

	cudaMemcpy(f,d_f,N*sizeof(float),cudaMemcpyDeviceToHost);
	for (int i=0; i < N; i++) {
		std::cout << i << " : " << f[i] << ", "; 
	}


	cudaFree(d_f);
	cudaDeviceReset();
	delete[] f;
	return 0;
}
