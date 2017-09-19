/**
 * @file   : simpleMathSoA_b.cu
 * @brief  : Measuring the vector summation kernel 
 * @details : A simple example of using a structure of arrays to store data on the device. 
 * 				This example is used to study the impact on performance of data layout on the 
 * 				GPU.  
 * @author : Ernest Yeung <ernestyalumni@gmail.com>
 * @date   : 20170915
 * @ref    : John Cheng, Max Grossman, Ty McKercher. Professional CUDA C Programming. 1st Ed. Wrox. 2014
 * 		   : Ch. 4 Global Memory; pp. 175 
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
 * nvcc -std=c++11 -arch='sm_52' simpleMathSoA.cu -o simpleMathSoA.exe
 * 
 * */
#include <iostream> 		
#include <vector>

struct SOA {
	float *x;
	float *y;
};

__global__ void initialize(SOA d_SOA) {
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	d_SOA.x[tid] = ((float) tid);
	d_SOA.y[tid] = 10.f * tid;	
}


__global__ void add(SOA d_SOA, const float a, const float b) {
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	d_SOA.x[tid] += a; 
	d_SOA.y[tid] += b;	
}



int main(int argc, char **argv) {
	int L = 64;
	
	int M = 4;
	int N = (L+M-1)/M;

	SOA d_SOA;
	cudaMallocManaged((void **) &d_SOA.x, sizeof(float)*L );
	cudaMallocManaged((void **) &d_SOA.y, sizeof(float)*L );


	initialize<<<N,M>>>(d_SOA) ;
	cudaDeviceSynchronize();

	SOA h_SOA; 
	h_SOA.x = (float *)malloc(sizeof(float)*L);
	h_SOA.y = (float *)malloc(sizeof(float)*L);


//	cudaMemcpy( h_SOA, d_SOA, sizeof(float)*2*L, cudaMemcpyHostToDevice);

	for (int idx=0;idx<8;idx++) { std::cout << d_SOA.x[idx] << " " << d_SOA.y[idx] << " "; } std::cout << std::endl; 
	cudaDeviceSynchronize();

	add<<<N,M>>>(d_SOA,11,333) ;
	cudaDeviceSynchronize();
	for (int idx=0;idx<8;idx++) { std::cout << d_SOA.x[idx] << " " << d_SOA.y[idx] << " "; } std::cout << std::endl; 

	cudaDeviceSynchronize();

	cudaMemcpy( h_SOA.x, d_SOA.x, sizeof(float)*L, cudaMemcpyDeviceToHost);
	cudaMemcpy( h_SOA.y, d_SOA.y, sizeof(float)*L, cudaMemcpyDeviceToHost);

	for (int idx=0;idx<L;idx++) { std::cout << h_SOA.x[idx] << " " << h_SOA.y[idx] << " "; } std::cout << std::endl; 


	cudaFree(d_SOA.x);
	cudaFree(d_SOA.y);

	free(h_SOA.x);
	free(h_SOA.y);


	cudaDeviceReset();
}
