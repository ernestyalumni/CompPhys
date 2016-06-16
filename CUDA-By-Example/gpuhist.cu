/* cf. Jason Sanders, Edward Kandrot. CUDA by Example: An Introduction to General-Purpose GPU Programming */
/* 
 * Chapter 9 Atomics 
 * 9.4 Computing Histograms
 * 9.4.1 CPU Histogram Computation
 */

#include "./common/hist.h"
#include "./common/errors.h"

#define SIZE (100*1024*1024)
 
__global__ void histo_kernel( unsigned char *buffer, 
							  long size, 
							  unsigned int *histo) { 
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	
	while (i < size) {
		atomicAdd( &(histo[buffer[i]]),1);
		i += stride;
	}
}
 
 
int main( void ) {
	unsigned char *buffer = (unsigned char*)big_random_block( SIZE );

	cudaEvent_t start, stop;
	HANDLE_ERROR(
		cudaEventCreate( &start ));
	HANDLE_ERROR(
		cudaEventCreate( &stop ));
	HANDLE_ERROR(
		cudaEventRecord( start, 0 ));

	// allocate memory on the GPU for the file's data
	unsigned char *dev_buffer;
	unsigned int  *dev_histo;
	HANDLE_ERROR(
		cudaMalloc( (void**)&dev_buffer, SIZE ));
	HANDLE_ERROR(
		cudaMemcpy( dev_buffer, buffer, SIZE, cudaMemcpyHostToDevice ));
	
	HANDLE_ERROR(
		cudaMalloc( (void**)&dev_histo, 256*sizeof(long) ));
	HANDLE_ERROR(
		cudaMemset( dev_histo, 0, 256 * sizeof( int ) ));
	// cudaMemset returns error code while memset doesn't
	
	cudaDeviceProp prop;
	HANDLE_ERROR( 
		cudaGetDeviceProperties( &prop, 0 ));
	int blocks = prop.multiProcessorCount;
	histo_kernel<<<blocks*2, 256>>>(dev_buffer,SIZE,dev_histo );
	

	unsigned int histo[256]; // each random 8-bit byte can be any of 256 different values
	HANDLE_ERROR( 
		cudaMemcpy( histo, dev_histo, 
					256 * sizeof( int ),
					cudaMemcpyDeviceToHost) );
					

	// get stop time, and display the timing results
	HANDLE_ERROR(
		cudaEventRecord( stop, 0 ));
	HANDLE_ERROR( 
		cudaEventSynchronize( stop ));
	float elapsedTime;
	HANDLE_ERROR( 
		cudaEventElapsedTime( &elapsedTime, 
								start, stop ));
	printf( "Time to generate:  %3.1f ms \n", elapsedTime );
		
		
	// sanity check	
	long histoCount = 0 ;
	for (int i=0; i<256; i++) {
		histoCount += histo[i];
	}
	printf( "Histogram Sum: %ld\n", histoCount);
	
	
	// verify that we have the same counts via CPU
	for (int i=0; i<SIZE; i++)
		histo[buffer[i]]--;
	for (int i=0; i<256; i++) {
		if (histo[i] != 0 )
			printf("Failure at %d! \n", i );
	}
	
	HANDLE_ERROR( cudaEventDestroy( start ) );
	HANDLE_ERROR( cudaEventDestroy( stop  ) );
	
	cudaFree( dev_histo );
	cudaFree( dev_buffer );
	
	free( buffer );
	return 0;
}
