/**
 * @file   : unifiedcoherency.cu
 * @brief  : Devices of compute capability 6.x on allow the CPUs and GPUs
 * 				to access Unified Memory allocations simultaneously via 
 * 				new page faulting mechanism
 * @details : Program can query whether device supports concurrent access to 
 * 				managed memory by checking a new concurrentManagedAccess property
 * @author : Ernest Yeung <ernestyalumni@gmail.com>
 * @date   : 20170913  
 * @ref    : http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-gpu-exclusive
 * 			: J.2.2.1. GPU Exclusive Access To Managed Memory
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
 * nvcc -arch='sm_52' unifiedcoherency.cu -o unifiedcoherency.exe
 * Note, I was on a GeForce GTX 980 Ti; the -arch='sm_52' was needed; otherwise, 
 * without it, I obtained these errors:
 *  error: __managed__ variables require architecture compute_30 or higher
 * */
__device__ __managed__ int x, y=2; 
__global__ void kernel() {
	x = 10;
}
int main() {
	kernel<<<1,1>>>();
//	y=20;	// Error on GPUs not supporting concurrent access
/*
 * Error obtained on GTX 980 Ti: 
 * Bus error (core dumped)
 * */
	
	cudaDeviceSynchronize();

	y=20; // Success on GPUs not supporting concurrent access
	return 0;
	
}
