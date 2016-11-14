/* dev_R2grid.cu
 * R3 under discretization (discretize functor) to a grid
 * Ernest Yeung  ernestyalumni@gmail.com
 * 20160728
 */
#include "dev_R2grid.h"

//__constant__ int dev_Ld[2];

__host__ dev_Grid2d::dev_Grid2d( dim3 Ld_in) : Ld(Ld_in)
{
	HANDLE_ERROR(
		cudaMalloc((void**)&this->dev_rho, this->NFLAT()*sizeof(float) ) );

	HANDLE_ERROR(
		cudaMalloc((void**)&this->dev_rho_out, this->NFLAT()*sizeof(float) ) );

	HANDLE_ERROR(
		cudaMalloc((void**)&this->dev_u, this->NFLAT()*sizeof(float2) ) );

	HANDLE_ERROR(
		cudaMalloc((void**)&this->dev_u_out, this->NFLAT()*sizeof(float2) ) );

	HANDLE_ERROR(
		cudaMalloc((void**)&this->dev_p, this->NFLAT()*sizeof(float2) ) );

	HANDLE_ERROR(
		cudaMalloc((void**)&this->dev_p_out, this->NFLAT()*sizeof(float2) ) );

	HANDLE_ERROR(
		cudaMalloc((void**)&this->dev_E, this->NFLAT()*sizeof(float) ) );

	HANDLE_ERROR(
		cudaMalloc((void**)&this->dev_E_out, this->NFLAT()*sizeof(float) ) );

}

__host__ dev_Grid2d::~dev_Grid2d() {
	HANDLE_ERROR(
		cudaFree( this->dev_rho ) );

	HANDLE_ERROR(
		cudaFree( this->dev_rho_out ) );

	HANDLE_ERROR(
		cudaFree( this->dev_u ) );

	HANDLE_ERROR(
		cudaFree( this->dev_u_out ) );

	HANDLE_ERROR(
		cudaFree( this->dev_p ) );

	HANDLE_ERROR(
		cudaFree( this->dev_p_out ) );

	HANDLE_ERROR(
		cudaFree( this->dev_E ) );

	HANDLE_ERROR(
		cudaFree( this->dev_E_out ) );
}


__host__ int dev_Grid2d :: NFLAT() {
	return Ld.x*Ld.y;
}	



