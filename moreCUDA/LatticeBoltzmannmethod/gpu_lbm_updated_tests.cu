/** 
 * gpu_lbm.cu
 * \file gpu_lbm.cu
 * Navier-Stokes equation solver in 2-dimensions, incompressible flow, by Lattice Boltzmann method
 * \brief Simulation of flow inside a 2D square cavity using the lattice Boltzmann method (LBM)
 * \author Ernest Yeung (I added improvements); I took a look at Joshi's code
 * Written by:  Abhijit Joshi (joshi1974@gmail.com)  
 * \email ernestyalumni@gmail.com
 * \date 20170112
 * 
 * cf. http://joshiscorner.com/files/src/blog/LBM-gpu-code.html
 * 
 * Compilation tips if you're not using a make file
 * 
 * nvcc -std=c++11 gpu_lbm.cu -o lbmGPU.exe
 *
 * Run instructions: ./lbmGPU.exe 
 */
 
#include <iostream>

#include "./commonlib/checkerror.h" // checkCudaErrors


	// the base vectors and weight coefficients (GPU)

__constant__ float2 dev_e[9]; // i = x,y; j = 0,1,...8

__constant__ float dev_alpha[9];

	// ant vector (GPU)
__constant__ int dev_ant[9];

	// END of the base vectors and weight coefficients (GPU)



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

}

__global__ void initialize( // FieldType *_ex, FieldType *_ey, FieldType *_alpha, 
//	int *_ant, 
	//FieldType *_ant, FieldType *_rh, FieldType *_ux, FieldType *_uy, FieldType *_f, FieldType *_feq, FieldType *_f_new) {
		
	float * rh , float2 *u, 
	float *f, float *feq, float *f_new,
	const int N_x, const int N_y, const int NDIR=9
	) {
	
	int i = threadIdx.x + blockIdx.x * blockDim.x ; 
	int j = threadIdx.y + blockIdx.y * blockDim.y ;
	
	if ((i >= N_x) || ( j >= N_y) ) { return; } // bound check for when accessing the array for array's values
	
	// initialize density and velocity fields inside the cavity; 
/*	rho[ i + N_x * j ] = DENSITY ; 
	u[ i + N_x * j].x = 0.0f ;
	u[ i + N_x * j].y = 0.0f;
	
	
	if (j == N_y-1 ) { u[ i + N_x * (N_y - 1) ] = LID_VELOCITY ; }
	*/
	
	// assign initial values for distribution functions
	int k = i + N_x * j ; 
	
	for (int dir = 0; dir < NDIR; dir++) {
		int index = i + j * N_x + dir * N_x * N_y ; 

		float edotu	= dev_e[dir].x * u[k].x + dev_e[dir].y * u[k].y ; 
		float udotu = u[k].x * u[k].x + u[k].y * u[k].y ;
		
		feq[index] = rh[k] * dev_alpha[dir] * ( 1.0f + 3.0f * edotu + 4.5f * edotu * edotu - 1.5f * udotu ) ; 
		f[index]   = feq[index];
		f_new[index] = feq[index] ; 
	}

}

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

int main( int argc, char *argv[])
{
	
	// problem parameters
	constexpr const int N_x = 512 ; 		// number of node points along x (cavity length in lattice units)
	constexpr const int N_y = 512 ; 		// number of node points along y (cavity length in lattice units)
	
	constexpr const int TIME_STEPS = 20000; 	// number of time steps for which the simulation is run
	
	constexpr const int NDIR = 9; 			// number of discrete velocity directions used in the D2Q9 model
	 
	constexpr const float DENSITY         = 2.7f ; 		// fluid density in lattice units
	constexpr const float LID_VELOCITY    = 0.05f;		// lid velocity in lattice units
	constexpr const float REYNOLDS_NUMBER = 100.0f; 	// Re=
		
	// END of problem parameters
		
		
			// initialize density and velocity fields inside the cavity; 
/*	rho[ i + N_x * j ] = DENSITY ; 
	u[ i + N_x * j].x = 0.0f ;
	u[ i + N_x * j].y = 0.0f;
	
	
	if (j == N_y-1 ) { u[ i + N_x * (N_y - 1) ] = LID_VELOCITY ; }
	*/

	// initialize density and velocity fields inside the cavity on host CPU
	float2 *u = new float2[N_x*N_y] ;
	float *rh = new float[N_x*N_y] ;

	for (int j = 0; j < N_y; ++j) { 
		for (int i = 0; i < N_x; ++i) {
			rh[ i + N_x*j]    = DENSITY ; 
			u[ i + N_x * j].x = 0.0f;
			u[ i + N_x * j].y = 0.0f;
		
		if (j == N_y-1) { u[ i + N_x * (N_y - 1)].x = LID_VELOCITY ; }
		}
	}
	
	
	// sanity check
/*
	std::cout << "\n Initially, on the host CPU : " << std::endl ; 
	std::cout << " d_rh : " << std::endl ;
	for (int j = N_y/2 - 21 ; j < N_y/2 + 21; j++) {
		for (int i = N_x/2 - 21; i < N_x/2 + 21; i++) {
			std::cout << rh[ i + N_x * j ] << " " ; 
		}
		std::cout << std::endl ;
	}

	std::cout << " d_u.x : " << std::endl ;
	for (int j = N_y/2 - 21 ; j < N_y/2 + 21; j++) {
		for (int i = N_x/2 - 21; i < N_x/2 + 21; i++) {
			std::cout << u[ i + N_x * j ].x << " " ; 
		}
		std::cout << std::endl ;
	}

	std::cout << " d_u.y : " << std::endl ;
	for (int j = N_y/2 - 21 ; j < N_y/2 + 21; j++) {
		for (int i = N_x/2 - 21; i < N_x/2 + 21; i++) {
			std::cout << u[ i + N_x * j ].y << " " ; 
		}
		std::cout << std::endl ;
	}
		
	// in the corner
	std::cout << " \n in the corner : " << std::endl;
	std::cout << " d_rh : " << std::endl ;
	for (int j = N_y - 20 ; j < N_y; j++) {
		for (int i = N_x - 20; i < N_x ; i++) {
			std::cout << rh[ i + N_x * j ] << " " ; 
		}
		std::cout << std::endl ;
	}

	std::cout << " d_u.x : " << std::endl ;
	for (int j = N_y - 20 ; j < N_y ; j++) {
		for (int i = N_x - 20; i < N_x ; i++) {
			std::cout << u[ i + N_x * j ].x << " " ; 
		}
		std::cout << std::endl ;
	}

	std::cout << " d_u.y : " << std::endl ;
	for (int j = N_y - 20 ; j < N_y ; j++) {
		for (int i = N_x - 20; i < N_x; i++) {
			std::cout << u[ i + N_x * j ].y << " " ; 
		}
		std::cout << std::endl ;
	}	
*/
	// END of sanity check
	
	
	
	
	
	// allocate memory on the GPU
	float *d_f, *d_feq, *d_f_new ;
	checkCudaErrors( 
		cudaMalloc((void **)&d_f, N_x * N_y * NDIR * sizeof(float)) );
	checkCudaErrors( 
		cudaMalloc((void **)&d_feq, N_x * N_y * NDIR * sizeof(float)) );
	checkCudaErrors( 
		cudaMalloc((void **)&d_f_new, N_x * N_y * NDIR * sizeof(float)) );
	
	float *d_rh ; 
	float2 *d_u; // velocity
	
	checkCudaErrors( 
		cudaMalloc((void **)&d_rh, N_x * N_y * sizeof(float)) );
	checkCudaErrors( 
		cudaMalloc((void **)&d_u, N_x * N_y * sizeof(float2)) );

	// set to 0
	checkCudaErrors( 
		cudaMemset(d_f, 0,N_x * N_y * NDIR * sizeof(float)) );
	checkCudaErrors( 
		cudaMemset(d_feq, 0,N_x * N_y * NDIR * sizeof(float)) );
	checkCudaErrors( 
		cudaMemset(d_f_new, 0,N_x * N_y * NDIR * sizeof(float)) );

	checkCudaErrors( 
		cudaMemset(d_rh, 0,N_x * N_y * sizeof(float)) );
	checkCudaErrors( 
		cudaMemset(d_u, 0,N_x * N_y * sizeof(float2)) );



	////////////////////////////////////////////////////////////////////
	// block, grid dimensions
	////////////////////////////////////////////////////////////////////
	// assign a 3D distribution of CUDA "threads" within each CUDA "block"
	int threadsAlongX = 32, threadsAlongY = 32;
	dim3 Blockdim( threadsAlongX , threadsAlongY, 1) ; 
	
	// calculate number of blocks along x and y in a 2D CUDA "grid"
	dim3 Griddim( (N_x + Blockdim.x -1)/Blockdim.x, (N_y + Blockdim.y -1)/Blockdim.y , 1) ;
	
	////////////////////////////////////////////////////////////////////
	// END of block, grid dimensions
	

	// initialize density and velocity fields inside the cavity on device GPU
	checkCudaErrors(
		cudaMemcpy( d_rh, rh, sizeof(float) * N_x*N_y, cudaMemcpyHostToDevice));

	checkCudaErrors(
		cudaMemcpy( d_u, u, sizeof(float2) * N_x*N_y, cudaMemcpyHostToDevice));
	
	
	// sanity check
/*
	checkCudaErrors(
		cudaMemcpy( rh, d_rh, sizeof(float) * N_x*N_y, cudaMemcpyDeviceToHost) );

	checkCudaErrors(
		cudaMemcpy( u, d_u, sizeof(float2) * N_x*N_y, cudaMemcpyDeviceToHost) );

	std::cout << " After Memcpy copy : " << std::endl;
	std::cout << " In the middle : " << std::endl ; 
	std::cout << " d_rh : " << std::endl ;
	for (int j = N_y/2 - 21 ; j < N_y/2 + 21; j++) {
		for (int i = N_x/2 - 21; i < N_x/2 + 21; i++) {
			std::cout << rh[ i + N_x * j ] << " " ; 
		}
		std::cout << std::endl ;
	}

	std::cout << " d_u.x : " << std::endl ;
	for (int j = N_y/2 - 21 ; j < N_y/2 + 21; j++) {
		for (int i = N_x/2 - 21; i < N_x/2 + 21; i++) {
			std::cout << u[ i + N_x * j ].x << " " ; 
		}
		std::cout << std::endl ;
	}

	std::cout << " d_u.y : " << std::endl ;
	for (int j = N_y/2 - 21 ; j < N_y/2 + 21; j++) {
		for (int i = N_x/2 - 21; i < N_x/2 + 21; i++) {
			std::cout << u[ i + N_x * j ].y << " " ; 
		}
		std::cout << std::endl ;
	}
		
	// in the corner
	std::cout << " \n in the corner : " << std::endl;
	std::cout << " d_rh : " << std::endl ;
	for (int j = N_y - 20 ; j < N_y; j++) {
		for (int i = N_x - 20; i < N_x ; i++) {
			std::cout << rh[ i + N_x * j ] << " " ; 
		}
		std::cout << std::endl ;
	}

	std::cout << " d_u.x : " << std::endl ;
	for (int j = N_y - 20 ; j < N_y ; j++) {
		for (int i = N_x - 20; i < N_x ; i++) {
			std::cout << u[ i + N_x * j ].x << " " ; 
		}
		std::cout << std::endl ;
	}

	std::cout << " d_u.y : " << std::endl ;
	for (int j = N_y - 20 ; j < N_y ; j++) {
		for (int i = N_x - 20; i < N_x; i++) {
			std::cout << u[ i + N_x * j ].y << " " ; 
		}
		std::cout << std::endl ;
	}	
	*/
	
	// END of sanity check
	
	
	
	
	
	set_e_alpha(); 
	
	initialize<<<Griddim, Blockdim>>>( d_rh, d_u, 
		d_f, d_feq, d_f_new, 
		N_x, N_y, NDIR ) ;
	
	
	// sanity check
/*
	checkCudaErrors(
		cudaMemcpy( rh, d_rh, sizeof(float) * N_x*N_y, cudaMemcpyDeviceToHost) );

	checkCudaErrors(
		cudaMemcpy( u, d_u, sizeof(float2) * N_x*N_y, cudaMemcpyDeviceToHost) );

	std::cout << " After initialize copy : " << std::endl;
	std::cout << " In the middle : " << std::endl ; 
	std::cout << " d_rh : " << std::endl ;
	for (int j = N_y/2 - 21 ; j < N_y/2 + 21; j++) {
		for (int i = N_x/2 - 21; i < N_x/2 + 21; i++) {
			std::cout << rh[ i + N_x * j ] << " " ; 
		}
		std::cout << std::endl ;
	}

	std::cout << " d_u.x : " << std::endl ;
	for (int j = N_y/2 - 21 ; j < N_y/2 + 21; j++) {
		for (int i = N_x/2 - 21; i < N_x/2 + 21; i++) {
			std::cout << u[ i + N_x * j ].x << " " ; 
		}
		std::cout << std::endl ;
	}

	std::cout << " d_u.y : " << std::endl ;
	for (int j = N_y/2 - 21 ; j < N_y/2 + 21; j++) {
		for (int i = N_x/2 - 21; i < N_x/2 + 21; i++) {
			std::cout << u[ i + N_x * j ].y << " " ; 
		}
		std::cout << std::endl ;
	}
		
	// in the corner
	std::cout << " \n in the corner : " << std::endl;
	std::cout << " d_rh : " << std::endl ;
	for (int j = N_y - 20 ; j < N_y; j++) {
		for (int i = N_x - 20; i < N_x ; i++) {
			std::cout << rh[ i + N_x * j ] << " " ; 
		}
		std::cout << std::endl ;
	}

	std::cout << " d_u.x : " << std::endl ;
	for (int j = N_y - 20 ; j < N_y ; j++) {
		for (int i = N_x - 20; i < N_x ; i++) {
			std::cout << u[ i + N_x * j ].x << " " ; 
		}
		std::cout << std::endl ;
	}

	std::cout << " d_u.y : " << std::endl ;
	for (int j = N_y - 20 ; j < N_y ; j++) {
		for (int i = N_x - 20; i < N_x; i++) {
			std::cout << u[ i + N_x * j ].y << " " ; 
		}
		std::cout << std::endl ;
	}	
	
	*/
	// END of sanity check
	
	
	
	
	// time integration
	int start_time = 0 ;
	for ( int t = start_time ; t < TIME_STEPS ; ++t) {
		
		timeIntegration<<<Griddim, Blockdim>>>( d_rh, d_u, 
			d_f, d_feq, d_f_new , 
			LID_VELOCITY, REYNOLDS_NUMBER, DENSITY, 
			N_x, N_y, NDIR ) ; 
	
	}
	
	// sanity check

	checkCudaErrors(
		cudaMemcpy( rh, d_rh, sizeof(float) * N_x*N_y, cudaMemcpyDeviceToHost) );

	checkCudaErrors(
		cudaMemcpy( u, d_u, sizeof(float2) * N_x*N_y, cudaMemcpyDeviceToHost) );

	std::cout << " After initialize and timeIntegration : " << std::endl; 
	std::cout << " d_rh : " << std::endl ;
	for (int j = N_y/2 - 21 ; j < N_y/2 + 21; j++) {
		for (int i = N_x/2 - 21; i < N_x/2 + 21; i++) {
			std::cout << rh[ i + N_x * j ] << " " ; 
		}
		std::cout << std::endl ;
	}

	std::cout << " d_u.x : " << std::endl ;
	for (int j = N_y/2 - 21 ; j < N_y/2 + 21; j++) {
		for (int i = N_x/2 - 21; i < N_x/2 + 21; i++) {
			std::cout << u[ i + N_x * j ].x << " " ; 
		}
		std::cout << std::endl ;
	}

	std::cout << " d_u.x : " << std::endl ;
	for (int j = N_y/2 - 21 ; j < N_y/2 + 21; j++) {
		for (int i = N_x/2 - 21; i < N_x/2 + 21; i++) {
			std::cout << u[ i + N_x * j ].y << " " ; 
		}
		std::cout << std::endl ;
	}
		
	// in the corner
	std::cout << " \n in the corner : " << std::endl;
	std::cout << " d_rh : " << std::endl ;
	for (int j = N_y - 20 ; j < N_y; j++) {
		for (int i = N_x - 20; i < N_x ; i++) {
			std::cout << rh[ i + N_x * j ] << " " ; 
		}
		std::cout << std::endl ;
	}

	std::cout << " d_u.x : " << std::endl ;
	for (int j = N_y - 20 ; j < N_y ; j++) {
		for (int i = N_x - 20; i < N_x ; i++) {
			std::cout << u[ i + N_x * j ].x << " " ; 
		}
		std::cout << std::endl ;
	}

	std::cout << " d_u.x : " << std::endl ;
	for (int j = N_y - 20 ; j < N_y ; j++) {
		for (int i = N_x - 20; i < N_x; i++) {
			std::cout << u[ i + N_x * j ].y << " " ; 
		}
		std::cout << std::endl ;
	}	
	
	
	// END of sanity check
	
	

	// free host CPU memory
	delete[] rh;
	delete[] u;

	// free device GPU memory
	checkCudaErrors(
		cudaFree( d_f ));
	checkCudaErrors(
		cudaFree( d_feq ));
	checkCudaErrors(
		cudaFree( d_f_new ));

	checkCudaErrors(
		cudaFree( d_rh ));
	checkCudaErrors(
		cudaFree( d_u ));

	return 0;
}






