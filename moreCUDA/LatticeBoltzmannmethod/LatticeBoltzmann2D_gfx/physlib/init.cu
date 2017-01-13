/** 
 * init.cu
 * \file init.cu
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
#include "init.h"

	// the base vectors and weight coefficients (GPU)

__constant__ float2 dev_e[9]; // i = x,y; j = 0,1,...8

__constant__ float dev_alpha[9];

	// ant vector (GPU)
__constant__ int dev_ant[9];

	// END of the base vectors and weight coefficients (GPU)


void set_u_0_CPU( std::vector<float2> & u, 
	const int N_x, const int N_y, const float LID_VELOCITY ) {

	for (int j = 0; j < N_y; ++j) { 
		for (int i = 0; i < N_x; ++i) {
			if (j == N_y-1) { u[ i + N_x * (N_y - 1)].x = LID_VELOCITY ; }
		}
	}

} // END of set_u_0_CPU

// set e, alpha
void set_e_alpha() {
	
	float2 *e = new float2[9] ;
	float *alpha = new float[9] ;
	int *ant = new int[9] ;

	
	e[0].x =  0.0; 	e[0].y =  0.0; 	alpha[0] = 4.0 / 9.0;
	e[1].x =  1.0; 	e[1].y =  0.0; 	alpha[1] = 1.0 / 9.0;
	e[2].x =  0.0; 	e[2].y =  1.0; 	alpha[2] = 1.0 / 9.0;
	e[3].x = -1.0; 	e[3].y =  0.0; 	alpha[3] = 1.0 / 9.0;
	e[4].x =  0.0; 	e[4].y = -1.0; 	alpha[4] = 1.0 / 9.0;
	e[5].x =  1.0; 	e[5].y =  1.0; 	alpha[5] = 1.0 / 36.0;
	e[6].x = -1.0; 	e[6].y =  1.0; 	alpha[6] = 1.0 / 36.0;
	e[7].x = -1.0; 	e[7].y = -1.0; 	alpha[7] = 1.0 / 36.0;
	e[8].x =  1.0; 	e[8].y = -1.0; 	alpha[8] = 1.0 / 36.0;
	
	ant[0] = 0; 	//		6 		2 		5
	ant[1] = 3; 	//		 		^ 		
	ant[2] = 4; 	//		 		| 		
	ant[3] = 1; 	//		 		| 		
	ant[4] = 2; 	//		3 <----	0 ----> 1	
	ant[5] = 7; 	//		 		| 		
	ant[6] = 8; 	//		 		| 		
	ant[7] = 5; 	//		 		V 		
	ant[8] = 6; 	//		7 		4 		8
	
	
	checkCudaErrors( 
		cudaMemcpyToSymbol( dev_e, e, sizeof(float2)*9, 0, cudaMemcpyHostToDevice) );  // offset from start is 0 
		
	checkCudaErrors( 
		cudaMemcpyToSymbol( dev_alpha, alpha, sizeof(float)*9, 0, cudaMemcpyHostToDevice) );  // offset from start is 0 

	checkCudaErrors( 
		cudaMemcpyToSymbol( dev_ant, ant, sizeof(int)*9, 0, cudaMemcpyHostToDevice) );  // offset from start is 0 


	delete[] e;
	delete[] alpha;
	delete[] ant;

} // END of set_e_alpha


// initialize
// doesn't work, obtained this error:  an illegal memory access was encountered cudaFree( this->f )
/*
__global__ void initialize( Dev_Grid2d & dev_grid2d ) {
	
	int i = threadIdx.x + blockIdx.x * blockDim.x ; 
	int j = threadIdx.y + blockIdx.y * blockDim.y ;

	const int N_x = dev_grid2d.Ld.x ;  
	const int N_y = dev_grid2d.Ld.y ;
	const int NDIR = dev_grid2d.NDIR ; 
	
	if ((i >= N_x) || ( j >= N_y) ) { return; } // bound check for when accessing the array for array's values
	
	
	// assign initial values for distribution functions
	int k = i + N_x * j ; 
	
	for (int dir = 0; dir < NDIR; dir++) {
		int index = i + j * N_x + dir * N_x * N_y ; 

		float edotu	= dev_e[dir].x * dev_grid2d.u[k].x + dev_e[dir].y * dev_grid2d.u[k].y ; 
		float udotu = dev_grid2d.u[k].x * dev_grid2d.u[k].x + 
							dev_grid2d.u[k].y * dev_grid2d.u[k].y ;
		
		float temp_feq = dev_grid2d.rh[k] * dev_alpha[dir] * 
			( 1.0f + 3.0f * edotu + 4.5f * edotu * edotu - 1.5f * udotu ) ; 

//		dev_grid2d.feq[index] = dev_grid2d.rh[k] * dev_alpha[dir] * 
//			( 1.0f + 3.0f * edotu + 4.5f * edotu * edotu - 1.5f * udotu ) ; 
		dev_grid2d.feq[index] = temp_feq; 

		dev_grid2d.f[index]   = temp_feq;
		dev_grid2d.f_new[index] = temp_feq ; 
	}

}
*/


// initialize, overloaded for "separate" function arguments
__global__ void initialize( 
	float * rh , float2 *u, 
	float *f, float *feq, float *f_new,
//	const int N_x, const int N_y, const int NDIR=9
	const int N_x, const int N_y, const int NDIR
	) {
		
	int i = threadIdx.x + blockIdx.x * blockDim.x ; 
	int j = threadIdx.y + blockIdx.y * blockDim.y ;
	
	if ((i >= N_x) || ( j >= N_y) ) { return; } // bound check for when accessing the array for array's values
	
	// assign initial values for distribution functions
	int k = i + N_x * j ; 
	
	for (int dir = 0; dir < NDIR; dir++) {
		int index = i + j * N_x + dir * N_x * N_y ; 

		float edotu	= dev_e[dir].x * u[k].x + dev_e[dir].y * u[k].y ; 
		float udotu = u[k].x * u[k].x + u[k].y * u[k].y ;
		
//		feq[index] = rh[k] * dev_alpha[dir] * ( 1.0f + 3.0f * edotu + 4.5f * edotu * edotu - 1.5f * udotu ) ; 
		float temp_feq = rh[k] * dev_alpha[dir] * ( 1.0f + 3.0f * edotu + 4.5f * edotu * edotu - 1.5f * udotu ) ; 
		feq[index] = temp_feq;
		f[index]   = temp_feq;
		f_new[index] = temp_feq ; 
	}
} // END of initialize, overloaded


