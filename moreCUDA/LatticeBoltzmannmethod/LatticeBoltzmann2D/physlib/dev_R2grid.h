/* dev_R2grid.h
 * R2 under discretization (discretize functor) to a (staggered) grid
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

class Dev_Grid2d
{
	public:
		dim3 Ld;  // Ld.x,Ld.y = L_x, L_y or i.e. imax,jmax 
		dim3 staggered_Ld; // Ld[0]+2,Ld[1]+2 = L_x+2,L_y+2, or i.e. imax+2,jmax+2 

		int NDIR;
		///////////////////////////////////////////////////////////////
		// Physical quantities over Euclidean space R^2, \mathbb{R}^2
		///////////////////////////////////////////////////////////////
		// 
		// rh
		// u2
		//
		// \rho \in C^{\infty}(\mathbb{R}^2) \xrightarrow{ \text{ discretize } }
		//  rh \in (\mathbb{R}^+)^{ Ld[0] * Ld[1] }
		//
		// and auxilliary float "fields" f, feq, f_new
		///////////////////////////////////////////////////////////////

		// float* for density, rh, and other scalar, auxilliary fields
			// allocate memory on the GPU
		float *rh ;
		float *f, *feq, *f_new ;

		// float2* for velocity
		float2* u; // velocity
	

		// Constructor
		/* --------------------------------------------------------- */
		/* Sets the initial values for velocity u, rh                 */
		/* --------------------------------------------------------- */
		__host__ Dev_Grid2d( dim3 , const int);

		// destructor
		__host__ ~Dev_Grid2d();

		__host__ int NFLAT();
		
		// __host__ int staggered_SIZE() - returns the staggered grid size
		/* this would correspond to Griebel's notation of 
		 * (imax+1)*(jmax+1)
		 */
		__host__ int staggered_SIZE();
		
		__host__ int flatten(const int i_x, const int i_y ) ;

		__host__ int staggered_flatten(const int i_x, const int i_y ) ;

};




#endif // __DEV_R2GRID_H__
