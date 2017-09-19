/**
 * @file   : simpleMathSoA.cu
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
 * nvcc -std=c++14 simpleMathSoA.cu -o simpleMathSoA.exe
 * 
 * */
#include <stdio.h> 		// printf

#define LEN 1<<22

struct InnerArray
{
	float x[LEN];
	float y[LEN];
};

struct InnerArray_gen
{
	float* x;
	float* y;
};

// functions for inner array out struct 
void initialInnerArray(InnerArray *ip, int size)
{
	for (int i = 0; i < size; i++) 
	{
		ip->x[i] = (float)( rand() & 0xFF ) / 100.0f; 
		ip->y[i] = (float)( rand() & 0xFF ) / 100.0f; 
	}

	return;
}

void initialInnerArray_gen(InnerArray_gen *ip, int size)
{
	for (int i = 0; i < size; i++) 
	{
		ip.x[i] = (float)( rand() & 0xFF ) / 100.0f; 
		ip.y[i] = (float)( rand() & 0xFF ) / 100.0f; 
	}

	return;
}


void printfHostResult(InnerArray *C, const int n)
{
	for (int idx = 0;idx < n; idx++)
	{
		printf("printout idx %d:  x %f y %f\n", idx, C->x[idx], C->y[idx]);
	}
	
	return;
}

void printfHostResult_gen(InnerArray_gen *C, const int n)
{
	for (int idx = 0;idx < n; idx++)
	{
		printf("printout idx %d:  x %f y %f\n", idx, C.x[idx], C.y[idx]);
	}
	
	return;
}

void checkInnerArray(Inner

