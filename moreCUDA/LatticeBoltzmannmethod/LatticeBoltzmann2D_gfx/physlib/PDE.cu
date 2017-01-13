/** 
 * PDE.cu
 * \file PDE.cu
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

#include "PDE.h"


__global__ void timeIntegration(
	float *rh, float2 *u, 
	float *f, float *feq, float *f_new,
	const float LID_VELOCITY, const float REYNOLDS_NUMBER , const float DENSITY,
	const int N_x, const int N_y, const int NDIR =9
	) {
	
	// calculate fluid viscosity based on the Reynolds number
	float kinematicViscosity = LID_VELOCITY * static_cast<float>( N_x ) / REYNOLDS_NUMBER;
	
	// calculate relaxation time tau
	float tau = 0.5f + 3.0f * kinematicViscosity ;
	
	// compute the "i" and "j" location and the "dir"
	// handle by this thread
	
	int i = threadIdx.x + blockIdx.x * blockDim.x ; 
	int j = threadIdx.y + blockIdx.y * blockDim.y ;

	int k = i + N_x * j; // k = 0, 1, ... blockDim.x * gridDim.x + N_x * blockDim.y * gridDim.y = N_x * N_y

	
	// collision
	if (( i >0 ) && ( i < N_x - 1) && (j > 0 ) && ( j < N_y - 1) ) { 
		for (int dir = 0; dir < NDIR; dir++ ) {
			
			int index = i + j * N_x + dir * N_x * N_y ; 

			float edotu	= dev_e[dir].x * u[k].x + dev_e[dir].y * u[k].y ; 
			float udotu = u[k].x * u[k].x + u[k].y * u[k].y ;
	
			feq[index] = rh[k] * dev_alpha[dir] * ( 1.0f + 3.0f * edotu + 4.5f * edotu * edotu - 1.5f * udotu ) ; 
		}
	}
	
	// sync/finish calculating feq (equilibrium)
	__syncthreads();
	
	// streaming from interior node points
	if ( (i > 0) && ( i < N_x - 1) && ( j > 0 ) && ( j < N_y - 1) ) { 
		for (int dir = 0; dir < NDIR; dir++ ) { 
			
			int index = i + j * N_x + dir * N_x * N_y ; // (i,j, dir)
			int index_new = ( i + dev_e[dir].x) + (j + dev_e[dir].y ) * N_x + dir*N_x*N_y ; 
	
			int index_ant = i + j * N_x + dev_ant[dir] * N_x * N_y ; 
			
			// post-collision distribution at (i,j) along "dir" 
			
			float f_plus = f[index] - ( f[index] - feq[index] ) / tau; 
			
			if (( i + dev_e[dir].x ==0) || ( i + dev_e[dir].x == N_x-1) || 
				( j + dev_e[dir].y == 0) || (j + dev_e[dir].y == N_y-1) ) {
				// bounce back
				
				int ixy = i + dev_e[dir].x + N_x * ( j + dev_e[dir].y ) ; 
				
				float ubdote = u[ixy].x * dev_e[dir].x + u[ixy].y * dev_e[dir].y ; 
				f_new[ index_ant ] = f_plus - 6.0 * DENSITY * dev_alpha[dir] * ubdote ; 
				
			}
			else { 
				// stream to neighbor
				f_new[index_new] = f_plus ;
			}
		}
	}
	
	// sync/finish calculating f_new (equilibrium)
	__syncthreads();
	
				
	// push f_new into f
	if (( i >0 ) && ( i < N_x - 1) && (j > 0 ) && (j < N_y - 1) ) {
		for (int dir=0; dir < NDIR; dir++) {
			int index = i + j * N_x + dir * N_x * N_y ;   	// (i,j,dir) 
			f[index] = f_new[index] ;
		}
	}
	
	// update density at interior nodes 
	if ((i>0) && (i <N_x-1) && (j >0) && (j< N_y -1)) {
		float rho = 0;
		for (int dir =0; dir < NDIR; dir++ ) {
			int index = i + j * N_x + dir * N_x * N_y ; 
			rho += f_new[index] ;
		}
		rh[k] = rho ;
	}
				
	// update velocity at interior nodes
	if ((i >0 ) && (i < N_x - 1) && (j >0 ) && (j < N_y -1) ) {
		float velx = 0.0f;
		float vely = 0.0f; 
		for (int dir = 0; dir < NDIR; dir++) {
			int index = i + j*N_x + dir* N_x * N_y ;
			velx += f_new[index] * dev_e[dir].x ; 
			vely += f_new[index] * dev_e[dir].y ;
			
		}
		
		u[k].x = velx / rh[k] ; 			
		u[k].y = vely / rh[k] ;
	}
} // END of time integration
