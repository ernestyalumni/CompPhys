/** 
 * init.h
 * \file init.h
 * Navier-Stokes equation solver in 2-dimensions, incompressible flow, by Lattice Boltzmann method
 * \brief initialization, set initial conditions, 
 * Simulation of flow inside a 2D square cavity using the lattice Boltzmann method (LBM)
 * \author Ernest Yeung; I had taken a look at Joshi's code
 * Abhijit Joshi (joshi1974@gmail.com)  
 * \email ernestyalumni@gmail.com
 * \date 20170112
 * 
 * cf. http://joshiscorner.com/files/src/blog/LBM-gpu-code.html
 * 
 * 
 *  
 * */
#ifndef __INIT_H__
#define __INIT_H__


#include <vector> // std::vector

#include "dev_R2grid.h" // Dev_R2grid
#include "../commonlib/checkerror.h"

	// the base vectors and weight coefficients (GPU)

extern __constant__ float2 dev_e[9]; // i = x,y; j = 0,1,...8

extern __constant__ float dev_alpha[9];

	// ant vector (GPU)
extern __constant__ int dev_ant[9];

	// END of the base vectors and weight coefficients (GPU)


void set_u_0_CPU( std::vector<float2> &, const int , const int , const float  ) ;


void set_e_alpha();

// initialize auxiliary, "distribution" functions

// doesn't work, obtained this error:  an illegal memory access was encountered cudaFree( this->f )

//__global__ void initialize( Dev_Grid2d & ) ; 

// initialize, overloaded, for "separate" function arguments
__global__ void initialize( 
	float * rh , float2 *u, 
	float *f, float *feq, float *f_new,
//	const int N_x, const int N_y, const int NDIR=9
	const int N_x, const int N_y, const int NDIR
	) ;




#endif // __INIT_H__
