/**
 * @file   : sumArraysOnHost.c
 * @brief  : Host-based array summation  
 * @author : Ernest Yeung	ernestyalumni@gmail.com
 * @date   : 20170911
 * @ref    : cf. John Cheng, Max Grossman, Ty McKercher.  Professional CUDA Programming.  
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
 * P.S. I'm using an EVGA GeForce GTX 980 Ti which has at most compute capability 5.2; 
 * A hardware donation or financial contribution would help with obtaining a 1080 Ti with compute capability 6.2
 * so that I can do -arch='sm_62' and use many new CUDA Unified Memory features
 * */
 /**
  * COMPILATION TIP(s)
  * (without make file, stand-alone)
  * gcc sumArraysOnHost.c -o sumArraysOnHost.exe
  * nvcc -Xcompiler -std=c99 sumArraysOnHost.c -o sum
  * -Xcompiler := specifies options directly to compiler/preprocessor 
  * */
#include <stdlib.h>
#include <string.h>
#include <time.h> 

void sumArraysOnHost(float *A, float *B, float *C, const int N) {
	for (int idx=0; idx<N; idx++) {
		C[idx] = A[idx] + B[idx]; 
	}
}

void initialData(float *ip, int size) {
	// generate different seed for random number
	time_t t;
	srand((unsigned int) time(&t));
	
	for (int i=0; i<size; i++) {
		ip[i] = (float) ( rand() & 0xFF )/10.0f; 
	}
}

int main(int argc, char **argv) {
	int nElem = 1024; 
	size_t nBytes = nElem * sizeof(float);
	
	float *h_A, *h_B, *h_C;
	h_A = (float *)malloc(nBytes); 
	h_B = (float *)malloc(nBytes); 
	h_C = (float *)malloc(nBytes); 
	
	initialData(h_A, nElem);
	initialData(h_B, nElem);
	
	sumArraysOnHost(h_A, h_B, h_C, nElem);
	
	free(h_A);
	free(h_B);
	free(h_C);
	
	return(0);
}


	
	
	

