/* dev_R2grid.cu
 * R3 under discretization (discretize functor) to a (staggered) grid
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
__host__ Dev_Grid2d::Dev_Grid2d( dim3 Ld_in, const int NDIR) : Ld(Ld_in), NDIR(NDIR)
{
	staggered_Ld.x  = Ld.x+2;
	staggered_Ld.y  = Ld.y+2;

	checkCudaErrors( 
		cudaMalloc((void **)&this->f, this->NFLAT() * NDIR * sizeof(float)) );
	checkCudaErrors( 
		cudaMalloc((void **)&this->feq, this->NFLAT() * NDIR * sizeof(float)) );
	checkCudaErrors( 
		cudaMalloc((void **)&this->f_new, this->NFLAT() * NDIR * sizeof(float)) );
	
	checkCudaErrors( 
		cudaMalloc((void **)&this->rh, this->NFLAT() * sizeof(float)) );
	checkCudaErrors( 
		cudaMalloc((void **)&this->u, this->NFLAT() * sizeof(float2)) );

	// set to 0
	checkCudaErrors( 
		cudaMemset(this->f, 0, this->NFLAT() * NDIR * sizeof(float)) );
	checkCudaErrors( 
		cudaMemset(this->feq, 0,this->NFLAT() * NDIR * sizeof(float)) );
	checkCudaErrors( 
		cudaMemset(this->f_new, 0,this->NFLAT() * NDIR * sizeof(float)) );

	checkCudaErrors( 
		cudaMemset(this->rh, 0,this->NFLAT() * sizeof(float)) );
	checkCudaErrors( 
		cudaMemset(this->u, 0,this->NFLAT() * sizeof(float2)) );

	
	
	
}

// destructor

__host__ Dev_Grid2d::~Dev_Grid2d() {

	// REMOVE this destructor (i.e.comment it out) when you want to use OpenGL graphics
	
	checkCudaErrors(
		cudaFree( this->f ));
	checkCudaErrors(
		cudaFree( this->feq ));
	checkCudaErrors(
		cudaFree( this->f_new ));

	checkCudaErrors(
		cudaFree( this->rh ));
	checkCudaErrors(
		cudaFree( this->u ));


}


__host__ int Dev_Grid2d :: NFLAT() {
	return Ld.x*Ld.y;
}	

__host__ int Dev_Grid2d :: staggered_SIZE() {
	return (staggered_Ld.x)*(staggered_Ld.y);
}	

__host__ int Dev_Grid2d :: flatten(const int i_x, const int i_y ) {
	return i_x+i_y*Ld.x  ;
}

__host__ int Dev_Grid2d :: staggered_flatten(const int i_x, const int i_y ) {
	return i_x+i_y*(staggered_Ld.x)  ;
}

