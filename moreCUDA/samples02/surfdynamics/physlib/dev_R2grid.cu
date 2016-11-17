/* dev_R2grid.cu
 * R3 under discretization (discretize functor) to a grid
 * Ernest Yeung  ernestyalumni@gmail.com
 * 2016115
 * 
 * compilation tip: (compile separately)
 * nvcc -std=c++11 -c ./physlib/dev_R2grid.cu -o dev_R2grid.o
 * 
 */
#include "dev_R2grid.h"

//__constant__ int dev_Ld[2];

// constructor
__host__ dev_Grid2d::dev_Grid2d( dim3 Ld_in) : Ld(Ld_in)
{
	checkCudaErrors(
		cudaMalloc((void**)&this->dev_f, this->NFLAT()*sizeof(float)) );

	checkCudaErrors(
		cudaMalloc((void**)&this->dev_f_out, this->NFLAT()*sizeof(float)) );


	checkCudaErrors(
		cudaMalloc((void**)&this->dev_u, this->NFLAT()*sizeof(float2)) );



	(this->channelDesc_f) = cudaCreateChannelDesc( 32, 0, 0, 0, 
													cudaChannelFormatKindFloat);

	// 8 bits * 4 bytes in float (sizeof(float)) = 32
/*	(this->channelDesc_f2) = cudaCreateChannelDesc( 32, 32, 0, 0, 
								cudaChannelFormatKindFloat);*/ // This gave a Segmentation Fault 
	(this->channelDesc_f2) = cudaCreateChannelDesc<float2>();
	

	checkCudaErrors(
		cudaMallocArray(&(this->cuArr_f), &(this->channelDesc_f), (this->Ld).x, (this->Ld).y, 
						cudaArraySurfaceLoadStore) ); 

	checkCudaErrors(
		cudaMallocArray(&(this->cuArr_f_out), &(this->channelDesc_f), (this->Ld).x, (this->Ld).y,
						cudaArraySurfaceLoadStore) ); 

	checkCudaErrors(
		cudaMallocArray(&(this->cuArr_u), &(this->channelDesc_f2), (this->Ld).x, (this->Ld).y,
						cudaArraySurfaceLoadStore) ); 

	checkCudaErrors(
		cudaMallocArray(&(this->cuArr_u_out), &(this->channelDesc_f2), (this->Ld).x, (this->Ld).y,
						cudaArraySurfaceLoadStore) ); 


}

// destructor

__host__ dev_Grid2d::~dev_Grid2d() {

	checkCudaErrors(
		cudaFree( this->dev_f ) );

	checkCudaErrors(
		cudaFree( this->dev_f_out ) );


	checkCudaErrors(
		cudaFree( this->dev_u ) );


	checkCudaErrors(
		cudaFreeArray( this->cuArr_f ));
		
	checkCudaErrors(
		cudaFreeArray( this->cuArr_f_out ));


	checkCudaErrors(
		cudaFreeArray( this->cuArr_u )); 

	checkCudaErrors(
		cudaFreeArray( this->cuArr_u_out )); 

}


__host__ int dev_Grid2d :: NFLAT() {
	return Ld.x*Ld.y;
}	



