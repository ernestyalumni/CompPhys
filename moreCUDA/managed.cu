/**
 * @file   : managed.cu
 * @brief  : __managed__ annotation to file-scope and global-scope CUDA __device__ variables
 
 * @author : Ernest Yeung <ernestyalumni@gmail.com>
 * @date   : 20170912  
 * @ref    : http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-gpu-exclusive
 * 			: J.2.1.2. Global-Scope Managed Variables Using __managed__
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
 * nvcc -arch='sm_51' managed.cu -o managed.exe
 * 
 * Here's where the -arch flag, for GPU architecture, gets really interesting and 
 * hardware specific.  
 * I have a GeForce GTX 980 Ti, if I do 
 * nvcc -arch='sm_51' managed.cu -o managed.exe
 * it compiles, but if I do 
 * nvcc -arch='sm_62' managed.cu -o managed.exe
 * it gives a Segmentation Fault.  Note that sm_62 works on Pascal architecture, such as a 1050
 * */
#include <stdio.h>

__device__ __managed__ int x[2];
__device__ __managed__ int y;
__global__ void kernel() {
	x[1] = x[0] + y;
}

int main() { 
	x[0] = 3;
	y = 5;
	kernel<<<1,1>>>();
	cudaDeviceSynchronize();
	printf("result = %d\n", x[1]);  // result = 8
	return 0;
}
