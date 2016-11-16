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
		cudaMalloc((void**)&this->dev_f_in, this->NFLAT()*sizeof(float) ) );

	checkCudaErrors(
		cudaMalloc((void**)&this->dev_f_out, this->NFLAT()*sizeof(float) ) );

	checkCudaErrors(
		cudaMalloc((void**)&this->dev_f_constsrc, this->NFLAT()*sizeof(float) ) );


	checkCudaErrors(
		cudaMalloc((void**)&this->dev_rho, this->NFLAT()*sizeof(float) ) );

	checkCudaErrors(
		cudaMalloc((void**)&this->dev_rho_out, this->NFLAT()*sizeof(float) ) );

	checkCudaErrors(
		cudaMalloc((void**)&this->dev_u, this->NFLAT()*sizeof(float2) ) );

	checkCudaErrors(
		cudaMalloc((void**)&this->dev_u_out, this->NFLAT()*sizeof(float2) ) );

	checkCudaErrors(
		cudaMalloc((void**)&this->dev_p, this->NFLAT()*sizeof(float2) ) );

	checkCudaErrors(
		cudaMalloc((void**)&this->dev_p_out, this->NFLAT()*sizeof(float2) ) );

	checkCudaErrors(
		cudaMalloc((void**)&this->dev_E, this->NFLAT()*sizeof(float) ) );

	checkCudaErrors(
		cudaMalloc((void**)&this->dev_E_out, this->NFLAT()*sizeof(float) ) );

	
}

// destructor
__host__ dev_Grid2d::~dev_Grid2d() {

	checkCudaErrors(
		cudaFree( this->dev_f_in ) );

	checkCudaErrors(
		cudaFree( this->dev_f_out ) );

	checkCudaErrors(
		cudaFree( this->dev_f_constsrc ) );


	checkCudaErrors(
		cudaFree( this->dev_rho ) );

	checkCudaErrors(
		cudaFree( this->dev_rho_out ) );

	checkCudaErrors(
		cudaFree( this->dev_u ) );

	checkCudaErrors(
		cudaFree( this->dev_u_out ) );

	checkCudaErrors(
		cudaFree( this->dev_p ) );

	checkCudaErrors(
		cudaFree( this->dev_p_out ) );

	checkCudaErrors(
		cudaFree( this->dev_E ) );

	checkCudaErrors(
		cudaFree( this->dev_E_out ) );


}


__host__ int dev_Grid2d :: NFLAT() {
	return Ld.x*Ld.y;
}	



