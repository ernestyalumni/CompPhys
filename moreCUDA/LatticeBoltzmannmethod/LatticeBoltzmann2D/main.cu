/** 
 * main.cu
 * \file gpu_lbm.cu
 * Navier-Stokes equation solver in 2-dimensions, incompressible flow, by Lattice Boltzmann method
 * \brief Simulation of flow inside a 2D square cavity using the lattice Boltzmann method (LBM)
 * \author Ernest Yeung; I had taken a look at Joshi's code
 * Abhijit Joshi (joshi1974@gmail.com)  
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

#include "./physlib/R2grid.h"      	// Grid2d
#include "./physlib/dev_R2grid.h"  	// Dev_Grid2d
#include "./physlib/init.h"			// set_u_0_CPU, dev_e[9], dev_alpha[9], dev_ant[9]
#include "./physlib/PDE.h"			// timeIntegration
 
int main(int argc, char* argv[]) {
	// ################################################################
	// ####################### Initialization #########################
	// ################################################################
	
	// problem parameters
	constexpr const int N_x = 2048 ; 		// number of node points along x (cavity length in lattice units)
	constexpr const int N_y = 2048 ; 		// number of node points along y (cavity length in lattice units)
	
	// 20000
	constexpr const int TIME_STEPS = 10; 	// number of time steps for which the simulation is run
	
	constexpr const int NDIR = 9; 			// number of discrete velocity directions used in the D2Q9 model
	 
	constexpr const float DENSITY         = 2.7f ; 		// fluid density in lattice units
	constexpr const float LID_VELOCITY    = 0.05f;		// lid velocity in lattice units
	constexpr const float REYNOLDS_NUMBER = 100.0f; 	// Re=
		
	// END of problem parameters
	
	
	// initialize density and velocity fields inside the cavity on host CPU
		// physics (on host); Euclidean (spatial) space
	constexpr std::array<int,2> LdS { N_x, N_y } ;
	constexpr std::array<float,2> ldS { 1.0, 1.0 };  // physical length

	Grid2d grid2d{LdS, ldS, 0.0f, 0.0f, DENSITY};  // u = v = 0 or u = (0,0) initial velocity
	
	set_u_0_CPU( grid2d.u2 , grid2d.Ld[0], grid2d.Ld[1], LID_VELOCITY );


	// initialize density and velocity fields inside the cavity on device GPU
	// physics (on device); Euclidean (spatial) space
	dim3 dev_L2 { static_cast<unsigned int>(N_x), 
					static_cast<unsigned int>(N_y) };

	Dev_Grid2d dev_grid2d( dev_L2, NDIR); 

	checkCudaErrors(
		cudaMemcpy( dev_grid2d.rh, grid2d.rh.data(), sizeof(float) * grid2d.NFLAT(), cudaMemcpyHostToDevice));

	checkCudaErrors(
		cudaMemcpy( dev_grid2d.u, grid2d.u2.data(), sizeof(float2) * grid2d.NFLAT(), cudaMemcpyHostToDevice));


	set_e_alpha();


	////////////////////////////////////////////////////////////////////
	// block, grid dimensions
	////////////////////////////////////////////////////////////////////
	// assign a 3D distribution of CUDA "threads" within each CUDA "block"
	// dim3 M_i where M_i.x = threadsAlongx, where M_i.y = threadsAlongy
	//
	///////////////////////////////////////////////////////////////////
	// Initialization (of f,feq,fnew) grid, block dimensions 
	
	dim3 init_M_i( 32 , 32, 1) ; 
	
	// calculate number of blocks along x and y in a 2D CUDA "grid"
	dim3 init_Griddim( (N_x + init_M_i.x -1)/init_M_i.x, (N_y + init_M_i.y -1)/init_M_i.y , 1) ;
	////////////////////////////////////////////////////////////////////
	// time integration grid, block dimensions
	dim3 M_i( 32 , 32, 1) ; 
	
	// calculate number of blocks along x and y in a 2D CUDA "grid"
	dim3 Griddim( (N_x + M_i.x -1)/M_i.x, (N_y + M_i.y -1)/M_i.y , 1) ;
	
	////////////////////////////////////////////////////////////////////
	// END of block, grid dimensions

	// initialize the auxiliary, so-called "distribution" functions on device GPU
	//initialize<<<init_Griddim, init_M_i>>>( dev_grid2d ) ; 

	initialize<<<init_Griddim, init_M_i>>>(dev_grid2d.rh, dev_grid2d.u, 
		dev_grid2d.f, dev_grid2d.feq, dev_grid2d.f_new, 
		N_x, N_y, NDIR ) ;


	// ################################################################
	// #######################               ##########################
	// #######################   MAIN LOOP   ##########################
	// #######################               ##########################
	// ################################################################
	
				/* t i m e    l o o p */
				/* ------------------ */
	// time integration
	int start_time = 0 ;
	for ( int t = start_time ; t < TIME_STEPS ; ++t) {

		timeIntegration<<<Griddim, M_i>>>( dev_grid2d.rh, dev_grid2d.u, 
			dev_grid2d.f, dev_grid2d.feq, dev_grid2d.f_new , 
			LID_VELOCITY, REYNOLDS_NUMBER, DENSITY, 
			N_x, N_y, NDIR ) ; 
	

	}  // END of time loop, time integration

	// ################################################################
	// ###################### END MAIN LOOP ###########################


// sanity check

	checkCudaErrors(
		cudaMemcpy( grid2d.rh.data(), dev_grid2d.rh, sizeof(float) * grid2d.NFLAT(), cudaMemcpyDeviceToHost));

	checkCudaErrors(
		cudaMemcpy( grid2d.u2.data(), dev_grid2d.u, sizeof(float2) * grid2d.NFLAT(), cudaMemcpyDeviceToHost));


	std::cout << "\n Initially, on the host CPU : " << std::endl ; 
	std::cout << " d_rh : " << std::endl ;
	for (int j = N_y/2 - 21 ; j < N_y/2 + 21; j++) {
		for (int i = N_x/2 - 21; i < N_x/2 + 21; i++) {
			std::cout << grid2d.rh[ i + N_x * j ] << " " ; 
		}
		std::cout << std::endl ;
	}

	std::cout << " d_u.x : " << std::endl ;
	for (int j = N_y/2 - 21 ; j < N_y/2 + 21; j++) {
		for (int i = N_x/2 - 21; i < N_x/2 + 21; i++) {
			std::cout << grid2d.u2[ i + N_x * j ].x << " " ; 
		}
		std::cout << std::endl ;
	}

	std::cout << " d_u.y : " << std::endl ;
	for (int j = N_y/2 - 21 ; j < N_y/2 + 21; j++) {
		for (int i = N_x/2 - 21; i < N_x/2 + 21; i++) {
			std::cout << grid2d.u2[ i + N_x * j ].y << " " ; 
		}
		std::cout << std::endl ;
	}
		
	// in the corner
	std::cout << " \n in the corner : " << std::endl;
	std::cout << " d_rh : " << std::endl ;
	for (int j = N_y - 20 ; j < N_y; j++) {
		for (int i = N_x - 20; i < N_x ; i++) {
			std::cout << grid2d.rh[ i + N_x * j ] << " " ; 
		}
		std::cout << std::endl ;
	}

	std::cout << " d_u.x : " << std::endl ;
	for (int j = N_y - 20 ; j < N_y ; j++) {
		for (int i = N_x - 20; i < N_x ; i++) {
			std::cout << grid2d.u2[ i + N_x * j ].x << " " ; 
		}
		std::cout << std::endl ;
	}

	std::cout << " d_u.y : " << std::endl ;
	for (int j = N_y - 20 ; j < N_y ; j++) {
		for (int i = N_x - 20; i < N_x; i++) {
			std::cout << grid2d.u2[ i + N_x * j ].y << " " ; 
		}
		std::cout << std::endl ;
	}	

	// END of sanity check






	// free device GPU memory
/*	checkCudaErrors(
		cudaFree( dev_grid2d.f ));
	checkCudaErrors(
		cudaFree( dev_grid2d.feq ));
	checkCudaErrors(
		cudaFree( dev_grid2d.f_new ));

	checkCudaErrors(
		cudaFree( dev_grid2d.rh ));
	checkCudaErrors(
		cudaFree( dev_grid2d.u ));
	*/
}  // END of main 
 




