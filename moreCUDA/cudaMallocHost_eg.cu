/**
 * @file   : cudaMallocHost_eg.cu
 * @brief  : Example of using cudaMallocHost 
 * @author : Ernest Yeung <ernestyalumni@gmail.com>
 * @date   : 20170909  
 * @ref    : https://github.com/monotone-RK/cuda-samples/blob/cc5f30f9cf4ad6c2766bfa05536cf7520ff57193/vecadd_simple/main.cu
 * @note   : On Fedora 23 Workstation, I sometimes get Segmentation Fault when N_0 below is
 * // 2048 (for 1<<12 or greater, I obtain a Segmentation Fault)
	// for a single float array, for size 1<<12 or greater, I obtain a Segmentation Fault
 * after exiting Sleep mode.  When I restart my computer, it doesn't have this problem anymore.  
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
 * nvcc cudaMallocHost_eg.cu -o cudaMallocHost.exe
 * 
 * */

#include <iostream>

__global__ void vecadd(float *a, float *b, float *c) {
	int tx = threadIdx.x + blockDim.x * blockIdx.x ;
	c[tx] = a[tx] + b[tx];
}

int main(int argc, char*argv[]) {
	const int N_0 = (1<<12) ; 
	std::cout << " N_0 : " << N_0 << std::endl; 
	
	float *a; 
	float *b; 
	float *c; 
	
	
	cudaMallocHost(&a, N_0*sizeof(float));
	cudaMallocHost(&b, N_0*sizeof(float));
	cudaMallocHost(&c, N_0*sizeof(float));

	
	for (int idx=0; idx<N_0; idx++) {
		a[idx] = 1.0f; 
		b[idx]  = 2.0f;
		c[idx] = 0.0f;
	}
	
	int M_x = (1<<5); 
	int N_x = (N_0+M_x-1)/M_x;
	vecadd<<<N_x,M_x>>>(a,b,c);
	
	for (int idx=0; idx<N_0; idx++) {
		std::cout << " idx : " << idx << ", c[idx] : " << c[idx] << std::endl; 
	}
	
	
	cudaFreeHost(a);
	cudaFreeHost(b);
	cudaFreeHost(c);
	
	return 0;
}
