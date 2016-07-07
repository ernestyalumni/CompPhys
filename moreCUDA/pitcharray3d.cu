/*
 * pitcharray3d.cu
 * 3-dimensional array allocation on device with cudaMalloc3d, demonstration
 * Ernest Yeung  ernestyalumni@gmail.com
 * 20160706
*/
#include <stdio.h> // printf
#include <iostream> // cout

__global__ void kernel(cudaPitchedPtr devPitchedPtr, int N_i, int N_j, int N_k)
{
	char* devPtr = (char *) devPitchedPtr.ptr;
	size_t pitch = devPitchedPtr.pitch;
	size_t slicePitch = pitch * N_j;

	for (int k = 0; k < N_k; ++k) {
		char* slice = devPtr + k * slicePitch;
		for (int j = 0; j < N_j; ++j) {
			float* row = (float*)(slice + j*pitch);
			for (int i = 0; i < N_i; ++i) {
				row[i] = powf( (float) i, (float) j) + ((float) k);
			}
		}
	}

}
	
__global__ void print_singl(cudaPitchedPtr devPitchedPtr, int i, int j, int k, int N_j)
{
	char* devPtr = (char *) devPitchedPtr.ptr;
	size_t pitch = devPitchedPtr.pitch;
	size_t slicePitch = pitch*N_j;

	char* slice = devPtr + k*slicePitch;
	float* row = (float*)(slice + j*pitch);
	printf( "%d, %d, %d : %3.1f \n", i,j,k, row[i]);
}

__global__ void print_partrow(cudaPitchedPtr devPitchedPtr, int j, int k, int N_j)
{
	char* devPtr = (char *) devPitchedPtr.ptr;
	size_t pitch = devPitchedPtr.pitch;
	size_t slicePitch = pitch*N_j;
	
	char* slice = devPtr + k*slicePitch;
	float* row = (float*)(slice + j*pitch);
	for (int i = 0 ; i < 10; ++i ) {
		printf( "%d, %d, %d : %3.1f ", i,j,k, row[i] );
	}
}

int main(int argc, char *argv[]) {
	int i {2}, j {2}, k {2};
/*	if (argc <= 1) {
		i = 2; 
		j = 2; 
		k = 2;
	}*/
	if (argc == 2) {
		i = atoi( argv[1] ) ;
	}
	else if (argc == 3) {
		i = atoi( argv[1] ) ; 
		j = atoi( argv[2] ) ;
	}
	else if (argc >= 4) {
		i = atoi( argv[1]) ; 
		j = atoi( argv[2] ) ; 
		k = atoi( argv[3] )  ;
	}
	
	cudaEvent_t start, stop; 

	cudaEventCreate( &start );
	cudaEventCreate( &stop );

	cudaEventRecord( start, 0);


	int N_i { 200 }, N_j { 200 }, N_k { 200 };

	cudaExtent extent = make_cudaExtent( N_i*sizeof(float), N_j, N_k);
	
	cudaPitchedPtr devPitchedPtr;
	cudaMalloc3D(&devPitchedPtr, extent);
	
	dim3 grid(32,1)  ; 
	dim3 block(16,1) ;
	
	kernel<<<grid,block>>>(devPitchedPtr, N_i,N_j,N_k);


	// Recording time for rough benchmarking
	cudaEventRecord( stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	std::cout << " \n Time in ms: " << elapsedTime << " ms" << std::endl;

	cudaEventRecord( start, 0);

	print_singl<<<1,1>>>(devPitchedPtr, i,j,k, N_j);

	cudaEventRecord( stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&elapsedTime, start, stop);
	std::cout << " \n Time in ms: " << elapsedTime << " ms" << std::endl;

	cudaEventRecord( start, 0);
	print_partrow<<<1,1>>>(devPitchedPtr, j,k, N_j);

	cudaEventRecord( stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&elapsedTime, start, stop);
	std::cout << " \n Time in ms: " << elapsedTime << " ms" << std::endl;


//	cudaFree( devPitchedPtr);
	
}
