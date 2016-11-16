/* dev_R2grid.h
 * R2 under discretization (discretize functor) to a thread block
 * Ernest Yeung  ernestyalumni@gmail.com
 * 20161114
 * 
 * compilation tip: (compile separately)
 * nvcc -std=c++11 -c ./physlib/dev_R2grid.cu -o dev_R2grid.o
 * 
 */
#ifndef __DEV_R2GRID_H__
#define __DEV_R2GRID_H__

#include "../commonlib/checkerror.h"  // checkCudaErrors

extern __constant__ int dev_Ld[2]; // L^{d=2} = (L_x,L_y) \in \mathbb{N}^2

class dev_Grid2d
{
	public:
		dim3 Ld;

		float *dev_f_in;
		float *dev_f_out;
		float *dev_f_constsrc;

		float *dev_rho;
		float *dev_rho_out;

		float2 *dev_u;
		float2 *dev_u_out;

		float2 *dev_p;
		float2 *dev_p_out;

		float *dev_E;
		float *dev_E_out;

		// constructor
		__host__ dev_Grid2d( dim3 );

		// destructor
		__host__ ~dev_Grid2d();

		__host__ int NFLAT();
};


#endif // __DEV_R2GRID_H__
