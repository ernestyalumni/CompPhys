/*
 * pitcharray2db.cu
 * 2-dimensional array allocation on device with pitch, pitched pointer, demonstration
 * Ernest Yeung  ernestyalumni@gmail.com
 * 20160706
*/
#include <iostream>
#include <stdio.h> // printf

__global__ void kernel(float* dev_array2d, size_t pitch, int N_i, int N_j)
{
	for (int i = 0 ; i < N_i; ++i)
	{
		// update the pointer to point to the beginning of the next row
		float* row = (float*)(((char*) dev_array2d) + (i * pitch));
		
		for (int j = 0 ; j < N_j; ++j ) {
			row[j] = powf( (float) j, (float) i);
		}
	}
	
	float* row1 = (float*)(((char*) dev_array2d) + (1 * pitch));
	// float* row1 = (float*)(((char*) dev_array2d) + (1 * pitch));
	for (int j = 0 ; j < 12 ; ++j) {
		printf(" %3.1f ", row1[j]);
	}

	float ele { ((float*)(((char*) dev_array2d)+(2*pitch)))[5] };
	printf("\n  %3.1f \n", ele );
}

int main() {

	cudaEvent_t start, stop; 

	cudaEventCreate( &start );
	cudaEventCreate( &stop );

	cudaEventRecord( start, 0);

	int N_j { 1000 };
	int N_i { 1000 };
	float* dev_array2d;
	
	size_t pitch;
	
	cudaMallocPitch(&dev_array2d, &pitch, N_j*sizeof(float), N_i);

	dim3 grid(2,1)  ; 
	dim3 block(2,1) ;
	
	kernel<<<grid,block>>>(dev_array2d, pitch, N_i,N_j);


	// Recording time for rough benchmarking
	cudaEventRecord( stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	std::cout << " \n Time in ms: " << elapsedTime << " ms" << std::endl;

	cudaFree( dev_array2d );
	
}
