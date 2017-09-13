/**
 * @file   : withoutunifiedmem.cu
 * @brief  : Simple program written without the benefit of unified memory:
 * 
 * @author : Ernest Yeung <ernestyalumni@gmail.com>
 * @date   : 20170912  
 * @ref    : http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-unified-memory-programming-hd
 * 			: J.1.1. Simplifying GPU Programming 
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
 * nvcc -std=c++11 withoutunifiedmem.cu -o withoutunifiedmem.exe
 * 
 * */
#include <stdio.h>

__global__ void AplusB(int *ret, int a, int b) {
	ret[threadIdx.x] = a+b+threadIdx.x;
}
int main() {
	int *ret;
	cudaMalloc(&ret, 1000 * sizeof(int));
	AplusB<<<1,1000>>>(ret,10,100);
	int *host_ret = (int *)malloc(1000 * sizeof(int));

	/*
	 * In non-managed example, synchronous cudaMemcpy() routine is used both 
	 * to synchronize the kernel (i.e. wait for it to finish running), &
	 * transfer data to host.  
	 */
	cudaMemcpy(host_ret, ret, 1000*sizeof(int), cudaMemcpyDefault);
	for (int i=0; i<1000;i++) {
		printf("%d: A+B = %d\n", i ,host_ret[i]);
	}
	free(host_ret);
	cudaFree(ret);
	return 0;
}
