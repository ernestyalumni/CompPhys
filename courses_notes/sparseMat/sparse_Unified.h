/**
 * @file   : sparse_Unified.h
 * @brief  : Sparse Matrices using CUDA Unified Memory, but for Compute Capability 5.X  
 * uses CUDA Unified Memory (Management)
 * @author : Ernest Yeung	ernestyalumni@gmail.com
 * @date   : 20170522
 * @ref    :  cf. https://www5.in.tum.de/lehre/vorlesungen/hpc/WS16/tutorial/sparse_02.pdf
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
 * P.S. I'm using an EVGA GeForce GTX 980 Ti which has at most compute capability 5.2; 
 * A hardware donation or financial contribution would help with obtaining a 1080 Ti with compute capability 6.2
 * so that I can do -arch='sm_62' and use many new CUDA Unified Memory features
 * */
/**
 * COMPILATION TIP
 * 
 * */

#ifndef __SPARSE_UNIFIED_H__
#define __SPARSE_UNIFIED_H__

#pragma once 

// CSR_MatVecMultiply - Compressed Sparse Matrix-Vector Multiply, i.e. Ax=y, matrix A, x,y vectors
/*
 * j = array of column indices
 * r_start[i] = r_start[i-1] + (number of nonzero elements in the (i-1)th original row in matrix A
 * N = number of rows in the matrix A
 * */
template <typename Type>
__global__ void CSR_MatVecMultiply(Type* a, Type* x, Type* y, int* j, int* r_start, const int N ) {
	
	int i = threadIdx.x + blockDim.x * blockIdx.x; // i is a row, let each thread compute a row
	if (i>=N) { return; } // check if thread has a row to compute
	
	Type yval = ((Type) 0.f);
	
	for (int k=r_start[i]; k<r_start[i+1]; k++) {
		yval += a[k] * x[j[k]]; 
	}
		
	y[i] = yval;
		
};

	

// CSR_MatVecMultiply_Warped - Compressed Sparse Matrix-Vector Multiply, i.e. Ax=y, matrix A, x,y vectors
/* 
 * Features -> 
 * 	warp coalescing
 * 
 * Definitions :
 * 	idx = 0,1,... N*WARPSIZE - 1, where N*WARPSIZE = number of rows * number of threads per row 
 * */
template <typename Type,int TILESIZE>
__global__ void CSR_MatVecMultiply_Warped(Type* a, Type* x, Type* y, int* j, int* r_start, const int N,
											const int WARPSIZE) {
	__shared__ Type vsh[TILESIZE];
	
	int idx = threadIdx.x + blockDim.x * blockIdx.x; // idx is a "global index", 
	int idx_warp = idx/WARPSIZE; 
	int idx_lane = idx & (WARPSIZE-1); // idx_lane = 0,1,...WARPSIZE-1
	int i = idx_warp; // i is effectively the row on the matrix that we're in

	if (i>=N) { return; } // check if warp has a row to compute or not 
	
	// Consider accumulating for each thread the running sum, i.e. compute running sum per thread
	vsh[threadIdx.x] = ((Type) 0.f);
	
	for (int k=r_start[i]+idx_lane; k<r_start[i+1]; k += WARPSIZE) {
		vsh[threadIdx.x] += a[k] * x[j[k]]; }
		
	// parallel reduction in shared memory
	for (int d = WARPSIZE >> 1; d>=1; d>>= 1) {
		if (idx_lane < d) {
			vsh[threadIdx.x] += vsh[threadIdx.x+d]; }
	}
	
	// 1st thread in a warp writes the result 
//	if (idx_warp == 0) {
	if (idx_lane == 0) {
		y[i] = vsh[threadIdx.x]; 
	}		
};
												
												
												

#endif // __SPARSE_UNIFIED_H__
