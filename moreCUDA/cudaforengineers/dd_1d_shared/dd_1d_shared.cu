/* dd_1d_shared.cu
 * 1-dimensional double derivative (dd for '') by finite difference with shared memory
 * Ernest Yeung  ernestyalumni@gmail.com
 * 20160625
 */
#include "dd_1d_shared.h"
#define M_x 64 // number of threads per block in x-direction
#define RAD 1 // radius of the stencil

__global__ void ddKernel(float *d_out, const float *d_in, int L_x, float h) {
	const int k_x = threadIdx.x + blockDim.x*blockIdx.x;
	if (k_x >= L_x) return;
	
	const int s_idx = threadIdx.x + RAD;
	
	extern __shared__ float s_in[];
	
	// Regular cells
	s_in[s_idx] = d_in[k_x];
	
	// Halo cells
	if (threadIdx.x < RAD) {
		s_in[s_idx - RAD] = d_in[k_x - RAD];
		s_in[s_idx + blockDim.x] = d_in[k_x+blockDim.x];
	}
	__syncthreads();
	
	d_out[k_x] = (s_in[s_idx-1]-2.f*s_in[s_idx]+s_in[s_idx+1])/(h*h);
}

void ddParallel(float *out, const float *in, int n, float h) {
	float *d_in = 0, *d_out = 0;
	
	cudaMalloc(&d_in, n*sizeof(float));
	cudaMalloc(&d_out, n*sizeof(float));
	cudaMemcpy(d_in, in, n*sizeof(float), cudaMemcpyHostToDevice);
	
//	ddKernel<<<(n + M_x - 1)/M_x, M_x,n*sizeof(float)>>>(d_out, d_in, n , h); // this line works
	ddKernel<<<(n + M_x - 1)/M_x, M_x,n*sizeof(float)>>>(d_out, d_in, n , h); // this line works

	
	cudaMemcpy(out, d_out, n*sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(d_in);
	cudaFree(d_out);
}
