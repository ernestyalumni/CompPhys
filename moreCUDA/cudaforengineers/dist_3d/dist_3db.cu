/* dist_3db.cu
 * 3-dim. Euclidean distance
 * Ernest Yeung  ernestyalumni@gmail.com
 * 20160625
 */
#include <stdio.h>
#include "../../common/errors.h"
#define Nthreads_x 1000 // total number of threads in x-direction (304 works)
#define Nthreads_y 1000 // total number of threads in y-direction (304 works)
#define Nthreads_z 1000 // total number of threads in z-direction (304 works)

#define M_x 20 // number of threads per block in x-direction
#define M_y 20 // number of threads per block in y-direction
#define M_z 20 // number of threads per block in z-direction

int blocksNeeded(int N_i, int M_i) { return (N_i+M_i-1)/M_i; }

__device__ float distance(int k_x, int k_y, int k_z, float3 x_0) {
	return sqrtf((k_x - x_0.x)*(k_x - x_0.x) + (k_y - x_0.y)*(k_y - x_0.y) + 
					(k_z - x_0.z)*(k_z - x_0.z));
	}

__global__ void distance(float *d_out, int L_x, int L_y, int L_z, float3 x_0) {
	// sanity check
//	printf("Hello thread x=%d,y=%d,z=%d \n", threadIdx.x, threadIdx.y,threadIdx.z);
	
	// map from threadIdx/blockIdx to (k_x,k_y,k_z) grid position
	const int k_x = threadIdx.x + blockIdx.x*blockDim.x;
	const int k_y = threadIdx.y + blockIdx.y*blockDim.y;
	const int k_z = threadIdx.z + blockIdx.z*blockDim.z;
	const int offset = k_x + k_y*L_x + k_z*L_x*L_y;
	if ((k_x >= L_x) || (k_y >= L_y) || (k_z >= L_z)) return;
	d_out[offset] = distance( k_x, k_y, k_z, x_0); // compute and store result

	// sanity check
//	printf("On global thread index x=%d,y=%d,z=%d, distance=%f\n",
//			k_x,k_y,k_z,d_out[offset]);
}
	
int main() {
	float *out = (float *)malloc(Nthreads_x*Nthreads_y*Nthreads_z*sizeof(float));
	float *d_out = 0;
	cudaMalloc(&d_out, Nthreads_x*Nthreads_y*Nthreads_z*sizeof(float));
	
	const float3 x_0 = { 0.0f, 0.0f, 0.0f }; // set reference position x_0
	const dim3 blockSize( M_x, M_y, M_z);
	const dim3 gridSize( blocksNeeded(Nthreads_x, M_x), blocksNeeded(Nthreads_y, M_y), 
						 blocksNeeded(Nthreads_z, M_z));
	distance<<<gridSize,blockSize>>>(d_out, Nthreads_x, Nthreads_y, Nthreads_z, x_0);
//	cudaDeviceSynchronize();
	cudaMemcpy(out, d_out, Nthreads_x*Nthreads_y*Nthreads_z*sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(d_out);

	// sanity check
	int testx = (int) 9.*Nthreads_x/10.;
	int testy = (int) 9.*Nthreads_y/10.;
	int testz = (int) 9.*Nthreads_z/10.;
	printf("At (%d,%d,%d), the distance is %f \n", testx, testy,testz,
			out[testx+testy*Nthreads_x+testz*Nthreads_x*Nthreads_y]); 

	free(out);
	return 0;
}
