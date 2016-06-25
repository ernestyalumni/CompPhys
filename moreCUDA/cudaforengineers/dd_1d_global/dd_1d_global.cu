/* dd_1d_global.cu
 * 1-dimensional double derivative (dd for '') by finite difference with global memory
 * Ernest Yeung  ernestyalumni@gmail.com
 * 20160625
 */
#include "dd_1d_global.h"
#define M_x 64 // number of threads per block in x-direction

__global__ void ddKernel(float *d_out, const float *d_in, int L_x, float h) {
	const int k_x = threadIdx.x + blockDim.x*blockIdx.x;
	if (k_x >= L_x) return;
	d_out[k_x] = (d_in[k_x-1]-2.f*d_in[k_x]+d_in[k_x+1])/(h*h);
}

void ddParallel(float *out, const float *in, int n, float h) {
	float *d_in = 0, *d_out = 0;
	
	cudaMalloc(&d_in, n*sizeof(float));
	cudaMalloc(&d_out, n*sizeof(float));
	cudaMemcpy(d_in, in, n*sizeof(float), cudaMemcpyHostToDevice);
	
	ddKernel<<<(n + M_x - 1)/M_x, M_x>>>(d_out, d_in, n , h);
	
	cudaMemcpy(out, d_out, n*sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(d_in);
	cudaFree(d_out);
}

