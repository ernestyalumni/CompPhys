/* dist_2d.cu
 * 2-dim. Euclidean distance
 * Ernest Yeung  ernestyalumni@gmail.com
 * 20160625
 */
#include <stdio.h>
#include "../common/errors.h"
#define L_x 512 // total number of threads in x-direction (480 works)
#define L_y 512 // total number of threads in y-direction (480 works)

#define N_x 16 // number of blocks in grid in x-direction (8 works)
#define N_y 16 // number of blocks in grid in y-direction (8 works)
#define M_x 4 // number of threads per block in x-direction (2 works)
#define M_y 4 // number of threads per block in y-direction (2 works)

int blocksNeeded(int N_i, int M_i) { return (N_i+M_i-1)/M_i; }

__device__ float distance(int k_x, int k_y, float2 x_0) {
	return sqrtf( (k_x - x_0.x)*(k_x - x_0.x) + (k_y - x_0.y)*(k_y - x_0.y) );
}

__global__ void distance(float *d_out, int l_x, int l_y, float2 x_0) {
	// sanity check
//	printf("Hello thread x=%d,y=%d,z=%d \n", threadIdx.x, threadIdx.y,threadIdx.z);
	
	// map from threadIdx/blockIdx to (k_x,k_y,k_z) grid position
	const int k_x = threadIdx.x + blockIdx.x*blockDim.x;
	const int k_y = threadIdx.y + blockIdx.y*blockDim.y;
	int offset = k_x + k_y*blockDim.x*gridDim.x ; 
//	d_out[offset] = distance( k_x,k_y,x_0);

	int maxlevelx = 1 + ((l_x-1)/(blockDim.x*gridDim.x));

	int levely = 0;
	int kprime_y = k_y+blockDim.y*gridDim.y*levely;

	while (kprime_y < l_y) {
		int levelx = 0;
		int kprime_x = k_x+blockDim.x*gridDim.x*levelx;
		d_out[offset] = distance(kprime_x,kprime_y,x_0);
		while (kprime_x < l_x) {
			kprime_x += blockDim.x*gridDim.x;
			levelx += 1;
			offset = (kprime_x-levelx*blockDim.x*gridDim.x)+
					 (kprime_y-levely*blockDim.y*gridDim.y)*blockDim.x*gridDim.x + 
					 levelx*blockDim.x*gridDim.x*blockDim.y*gridDim.y + 
					 levely*blockDim.x*gridDim.x*blockDim.y*gridDim.y*maxlevelx;
			d_out[offset] = distance(kprime_x,kprime_y,x_0);		 
		}
		kprime_y += blockDim.y*gridDim.y;
		levely += 1;

		__syncthreads();
	}
	__syncthreads();
}
	
	
int main() {
	float *out = (float *)malloc(L_x*L_y*sizeof(float));
	float *d_out = 0;
	cudaMalloc(&d_out, L_x*L_y*sizeof(float));
	
	const float2 x_0 = { 0.0f, 0.0f }; // set reference position x_0
	const dim3 blockSize( M_x, M_y);
	// blocksNeeded function isn't really needed
	const dim3 gridSize( blocksNeeded(N_x*M_x, M_x), blocksNeeded(N_y*M_y, M_y));
	distance<<<gridSize,blockSize>>>(d_out, L_x, L_y, x_0);
	cudaDeviceSynchronize();
	cudaMemcpy(out, d_out, L_x*L_y*sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(d_out);

	// sanity check
	int testx = (int) 1.*L_x/10.;
	int testy = (int) 1.*L_y/10.;
	printf("At (%d,%d), the distance is %f \n", testx, testy,
			out[testx+testy*N_x*M_x]); 

	int maxlevelx = 1 + ((L_x-1)/(N_x*M_x));

	testx = (int) 9.*L_x/10.;
	testy = (int) 9.*L_y/10.;
	int testlevelx = testx/(N_x*M_x);
	int testlevely = testy/(N_y*M_y);
	printf("At (%d,%d), the distance is %f \n", testx, testy,
			out[ testx-testlevelx*N_x*M_x+
				 (testy-testlevely*N_y*M_y)*N_x*M_x+
				 testlevelx*N_x*M_x*N_y*M_y+ 
				 testlevely*N_x*M_x*N_y*M_y*maxlevelx]);
				 	
	free(out);
	return 0;
}

