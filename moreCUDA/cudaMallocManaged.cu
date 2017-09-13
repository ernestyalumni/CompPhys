/**
 * @file   : cudaMallocManaged.cu
 * @brief  : simple example, showing use of cudaMallocManaged
 * @author : Ernest Yeung <ernestyalumni@gmail.com>
 * @date   : 20170912  
 * @ref    : http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-explicit-allocation
 * 			: J.2.1.1. Explicit Allocation Using cudaMallocManaged
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
 * nvcc cudaMallocManaged.cu -o cudaMallocManaged.exe
 * 
 * */
#include <stdio.h>

__global__ void printme(char *str) {
	printf(str);
}

int main() {
	// Allocate 100 bytes of memory, accessible to both Host and 
	// Device code 
	char *s;
	cudaMallocManaged(&s, 100);
	// Note direct Host-code use of "s"
	strncpy(s, "Hello Unified Memory\n", 99);
	// Here we pass "s" to a kernel without explicitly copying
	printme<<<1,1>>>(s);
	cudaDeviceSynchronize();
	// Free as for normal CUDA allocations
	cudaFree(s);
	return 0;
}
