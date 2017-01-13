/** 
 * PDE.h
 * \file PDE.h
 * Navier-Stokes equation solver in 2-dimensions, incompressible flow, by Lattice Boltzmann method
 * \brief PDE, partial differential equation, dynamics 
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

#ifndef __PDE_H__
#define __PDE_H__

	// the base vectors and weight coefficients (GPU)
#include "init.h" // dev_e, dev_alpha, dev_ant


__global__ void timeIntegration(
	float *rh, float2 *u, 
	float *f, float *feq, float *f_new,
	const float LID_VELOCITY, const float REYNOLDS_NUMBER , const float DENSITY,
	const int N_x, const int N_y, const int NDIR 
	) ;

#endif // __PDE_H__
