/* surfdynamics2d.cu
 * surface dynamics, in 2-dimensions
 * Ernest Yeung  ernestyalumni@gmail.com
 * 20161116
 * 
 * Compilation tips if you're not using a make file
 * 
 * nvcc -std=c++11 -c ./physlib/R2grid.cpp -o R2grid.o  // or 
 * g++ -std=c++11 -c ./physlib/R2grid.cpp -o R2grid.o
 * 
 * nvcc -std=c++11 -c ./physlib/dev_R2grid.cu -o dev_R2grid.o
 * nvcc -std=c++11 -c ./commonlib/surfObj2d.cu -o surfObj2d.o
 * nvcc -std=c++11 surfdynamics2d.cu R2grid.o dev_R2grid.o surfObj2d.o -o surfdynamics2d
 * 
 */
#include <iostream> // std::cout
#include <fstream>  // std::ofstream

#include "./physlib/R2grid.h" // Grid2d
#include "./physlib/dev_R2grid.h" // dev_Grid2d
#include "./commonlib/surfObj2d.h" // SurfObj2d
#include "./commonlib/checkerror.h" // checkCudaErrors

constexpr const int L_X  { 128 } ;  // WIDTH
constexpr const int L_Y { 64 } ;    // HEIGHT

// Simple copy kernel
/* http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#surface-object-api
 * 3.2.11.2.1. Surface Object API */
__global__ void copyKernel( cudaSurfaceObject_t inputSurfObj,
							cudaSurfaceObject_t outputSurfObj,
							int width, int height)
{
	// Calculate surface coordinates
	const int k_x = threadIdx.x + blockIdx.x * blockDim.x ; 
	const int k_y = threadIdx.y + blockIdx.y * blockDim.y ; 
	if ((k_x >= width) || (k_y >= height)) {
		return; }
		
	float data;
	// Read from input surface
	surf2Dread(&data, inputSurfObj, k_x * 4, k_y, cudaBoundaryModeClamp );
//	surf2Dread(&data, inputSurfObj, (k_x+1) * 4, k_y, cudaBoundaryModeClamp );
	surf2Dwrite(data, outputSurfObj, k_x * 4, k_y );		

}

__global__ void addrKernel( cudaSurfaceObject_t inputSurfObj, 
							cudaSurfaceObject_t outputSurfObj,
							int L_x, int L_y) {
	// Calculate surface coordinates
	const int k_x = threadIdx.x + blockIdx.x * blockDim.x ;
	const int k_y = threadIdx.y + blockIdx.y * blockDim.y ;
	if ((k_x >= L_x) || (k_y >= L_y)) { 
		return ; }
		
	float c, r, tempval;
	surf2Dread(&c, inputSurfObj, k_x * 4, k_y ) ; 
	surf2Dread(&r, inputSurfObj, (k_x+1) * 4, k_y , cudaBoundaryModeClamp) ; 
	
	tempval = c+r;
	surf2Dwrite( tempval, outputSurfObj, k_x * 4, k_y); 
								
}

__global__ void addlKernel( cudaSurfaceObject_t inputSurfObj, 
							cudaSurfaceObject_t outputSurfObj,
							int L_x, int L_y) {
	// Calculate surface coordinates
	const int k_x = threadIdx.x + blockIdx.x * blockDim.x ;
	const int k_y = threadIdx.y + blockIdx.y * blockDim.y ;
	if ((k_x >= L_x) || (k_y >= L_y)) { 
		return ; }
		
	float c, l, tempval;
	surf2Dread(&c, inputSurfObj, k_x * 4, k_y ) ; 
	surf2Dread(&l, inputSurfObj, (k_x-1) * 4, k_y , cudaBoundaryModeClamp) ; 
	
	tempval = c+l;
	surf2Dwrite( tempval, outputSurfObj, k_x * 4, k_y); 
								
}

void addlrKernels_launch( cudaSurfaceObject_t inputSurfObj, 
							cudaSurfaceObject_t outputSurfObj,
							const int L_x, const int L_y, 
							dim3 M_in, const int NITERS =1) {
	volatile bool dstOut = true; 

	dim3 dimGrid((L_x + M_in.x - 1)/ M_in.x , 
					(L_y + M_in.y - 1) / M_in.y);

	for (auto iter = 0; iter < NITERS; ++iter) {
		cudaSurfaceObject_t inSurfObj, outSurfObj; 
		if (dstOut) {
			inSurfObj  = inputSurfObj ;
			outSurfObj = outputSurfObj   ;
			addrKernel<<<dimGrid,M_in>>>( inSurfObj, outSurfObj, L_x,L_y) ; 
		}
		else {
			outSurfObj = inputSurfObj; 
			inSurfObj  = outputSurfObj;
			addlKernel<<<dimGrid,M_in>>>( inSurfObj, outSurfObj, L_x,L_y) ; 
		}
		dstOut = !dstOut;
	}
}

// addrxyf2Kernel - add the right element and do it for both x,y components for float2
__global__ void addrxyf2Kernel( cudaSurfaceObject_t inputSurfObj, 
							cudaSurfaceObject_t outputSurfObj,
							int L_x, int L_y) {
	// Calculate surface coordinates
	const int k_x = threadIdx.x + blockIdx.x * blockDim.x ;
	const int k_y = threadIdx.y + blockIdx.y * blockDim.y ;
	if ((k_x >= L_x) || (k_y >= L_y)) { 
		return ; }

	float2 c, r, tempval;

	const int RADIUS = 1;
	int stencilindex_x = k_x + RADIUS;
	stencilindex_x = min( max( stencilindex_x,0), L_x-1) ;
	surf2Dread(&c, inputSurfObj, k_x * 4, k_y ) ; 
	surf2Dread(&r, inputSurfObj, ( stencilindex_x ) * 4, k_y , cudaBoundaryModeClamp) ; 
	
//	tempval = c+r;
	tempval.x = c.x + r.x ;
	tempval.y = c.y + r.y ; 
	surf2Dwrite( tempval, outputSurfObj, k_x * 4, k_y); 
								
}

int main(int argc, char* argv[]) {
	// sanity check - surface memory read/writes use byte-addressing, so what's the number of bytes of a float?
	std::cout << "\n sizeof(float) : " << sizeof(float) << std::endl ;

	// boilerplate
//	constexpr const int DISPLAY_SIZE { 14 };
	constexpr const int NITERS { 2 };
	
	// physics; Euclidean space
	constexpr std::array<int,2> LdS { L_X, L_Y };
	constexpr std::array<float,2> ldS { 1.f , 1.f };
	
	Grid2d grid2d{ LdS, ldS };
	
	dim3 dev_L2 { static_cast<unsigned int>(L_X),
					static_cast<unsigned int>(L_Y) };
	dev_Grid2d dev_grid2d( dev_L2 );
	
	// initial condition
	for (auto j = 0; j < grid2d.Ld[1] ; ++j) { 
		for (auto i = 0; i < grid2d.Ld[0]; ++i ) {
			grid2d.f[ grid2d.flatten(i,j) ] = static_cast<float>( grid2d.flatten(i,j) ) + 0.1f ; 
		}
	}
	
	// Copy to device memory some data located at address grid2d.rho in host memory
	checkCudaErrors( 
		cudaMemcpy(dev_grid2d.dev_f, 
						(grid2d.f).data(), sizeof(float)*grid2d.NFLAT(), cudaMemcpyHostToDevice) );
	
	checkCudaErrors( 
		cudaMemcpyToArray(dev_grid2d.cuArr_f, 0, 0,
						dev_grid2d.dev_f, sizeof(float)*dev_grid2d.NFLAT(), cudaMemcpyDeviceToDevice) );
	
	
	// Create surface object
	SurfObj2d surf_f( dev_grid2d.cuArr_f ) ;
	SurfObj2d surf_f_out( dev_grid2d.cuArr_f_out ) ;


	// Invoke kernel
	// MANUALLY CHANGE M_i here
	constexpr const int M_X { 8 }; 
	constexpr const int M_Y { 4 };

	dim3 M_i(M_X,M_Y);
	dim3 dimGrid((L_X + M_i.x - 1)/ M_i.x , 
					(L_Y + M_i.y - 1) / M_i.y);

	copyKernel<<<dimGrid, M_i>>>( surf_f.surfObj, 
									surf_f_out.surfObj,
										L_X,L_Y);
					
	// copy result, output array from device to host memory				
	checkCudaErrors(
		cudaMemcpyFromArray( grid2d.f_out.data(), dev_grid2d.cuArr_f_out,
							0,0, dev_grid2d.NFLAT()*sizeof(float), cudaMemcpyDeviceToHost) );

	// C++ file Input/Output <fstream>
	std::ofstream simplecpy_file;
	simplecpy_file.open("./dataout/simplecpy.csv");
	for (auto j=0; j<grid2d.Ld[1]; ++j) {
		// print first column's element
		simplecpy_file << grid2d.f_out[ 0 + j * grid2d.Ld[0] ] ;
		
		// print remaining columns
		for (auto i=1; i<grid2d.Ld[0]; ++i) {
			simplecpy_file << ", " << grid2d.f_out[i+j*grid2d.Ld[0] ] ;
		}
		
		// print new line between rows
		simplecpy_file << std::endl;
	}
	simplecpy_file.close();

	
	// testing out once (1) addrKernel
	addrKernel<<<dimGrid, M_i>>>( surf_f.surfObj, 
									surf_f_out.surfObj,
										L_X,L_Y);
	// copy result, output array from device to host memory				
	checkCudaErrors(
		cudaMemcpyFromArray( grid2d.f_out.data(), dev_grid2d.cuArr_f_out,
							0,0, dev_grid2d.NFLAT()*sizeof(float), cudaMemcpyDeviceToHost) );
	// C++ file Input/Output <fstream>
	std::ofstream addr01_file;
	addr01_file.open("./dataout/addr01.csv");
	for (auto j=0; j<grid2d.Ld[1]; ++j) {
		// print first column's element
		addr01_file << grid2d.f_out[ 0 + j * grid2d.Ld[0] ] ;
		
		// print remaining columns
		for (auto i=1; i<grid2d.Ld[0]; ++i) {
			addr01_file << ", " << grid2d.f_out[i+j*grid2d.Ld[0] ] ;
		}
		
		// print new line between rows
		addr01_file << std::endl;
	}
	addr01_file.close();

	
	// testing out addlrKernels_launch
	addlrKernels_launch( surf_f.surfObj, surf_f_out.surfObj, L_X, L_Y, M_i, NITERS ) ;
	// copy results, output arrays from device to host memory				
	checkCudaErrors(
		cudaMemcpyFromArray( grid2d.f.data(), dev_grid2d.cuArr_f,
							0,0, dev_grid2d.NFLAT()*sizeof(float), cudaMemcpyDeviceToHost) );
	checkCudaErrors(
		cudaMemcpyFromArray( grid2d.f_out.data(), dev_grid2d.cuArr_f_out,
							0,0, dev_grid2d.NFLAT()*sizeof(float), cudaMemcpyDeviceToHost) );
	// C++ file Input/Output <fstream>
	std::ofstream addlrin_file, addlrout_file ;
	addlrin_file.open("./dataout/addlrin.csv");
	addlrout_file.open("./dataout/addlrout.csv");
	for (auto j=0; j<grid2d.Ld[1]; ++j) {
		// print first column's element
		addlrin_file   << grid2d.f[ 0 + j * grid2d.Ld[0] ] ;
		addlrout_file << grid2d.f_out[ 0 + j * grid2d.Ld[0] ] ;
		
		// print remaining columns
		for (auto i=1; i<grid2d.Ld[0]; ++i) {
			addlrin_file  << ", " << grid2d.f[i+j*grid2d.Ld[0] ] ;
			addlrout_file << ", " << grid2d.f_out[i+j*grid2d.Ld[0] ] ;
		}
		
		// print new line between rows
		addlrin_file  << std::endl;
		addlrout_file << std::endl;
	}
	addlrin_file.close();
	addlrout_file.close();
	
	
	// Create surface object to test the "binding" to the same cudaArray
	SurfObj2d surf_p( dev_grid2d.cuArr_f ) ;
	// testing out once (1) addrKernel
	addrKernel<<<dimGrid, M_i>>>( surf_p.surfObj, 
									surf_f.surfObj,
										L_X,L_Y);
	// copy results, output arrays from device to host memory				
	checkCudaErrors(
		cudaMemcpyFromArray( grid2d.f.data(), dev_grid2d.cuArr_f,
							0,0, dev_grid2d.NFLAT()*sizeof(float), cudaMemcpyDeviceToHost) );
	// C++ file Input/Output <fstream>
	std::ofstream addr_p_file ;
	addr_p_file.open("./dataout/addr_p.csv");
	for (auto j=0; j<grid2d.Ld[1]; ++j) {
		// print first column's element
		addr_p_file   << grid2d.f[ 0 + j * grid2d.Ld[0] ] ;
		
		// print remaining columns
		for (auto i=1; i<grid2d.Ld[0]; ++i) {
			addr_p_file  << ", " << grid2d.f[i+j*grid2d.Ld[0] ] ;
		}
		
		// print new line between rows
		addr_p_file  << std::endl;
	}
	addr_p_file.close();
	
	
	// initial condition for u
	for (auto j = 0; j < grid2d.Ld[1] ; ++j) { 
		for (auto i = 0; i < grid2d.Ld[0]; ++i ) {
			grid2d.u[ grid2d.flatten(i,j) ].x = static_cast<float>( grid2d.flatten(i,j))*10.f+0.1f;
			grid2d.u[ grid2d.flatten(i,j) ].y = static_cast<float>( grid2d.flatten(i,j))*0.001f + 0.00001f ; 
		}
	}
	
	
	// Copy to device memory some data located at address grid2d.u in host memory
	checkCudaErrors( 
		cudaMemcpyToArray(dev_grid2d.cuArr_u, 0, 0,
						(grid2d.u).data(), sizeof(float2)*dev_grid2d.NFLAT(), cudaMemcpyHostToDevice) );

	// Create surface object to test the "binding" to cudaArray for float2
	SurfObj2d surf_u( dev_grid2d.cuArr_u ) ;
	SurfObj2d surf_u_out( dev_grid2d.cuArr_u_out ) ;
	// Invoke kernel
	// testing out once (1) addrxyKernel
/*
 * misaligned address 
	addrxyf2Kernel<<<dimGrid, M_i>>>( surf_u.surfObj, 
									surf_u_out.surfObj,
										L_X,L_Y);
									 
*/										
	// copy results, output arrays from device to host memory				
/*	checkCudaErrors(
		cudaMemcpyFromArray( (grid2d.u).data(), dev_grid2d.cuArr_u,
							0,0, dev_grid2d.NFLAT()*sizeof(float2), cudaMemcpyDeviceToHost) );

misaligned address cudaMemcpyFromArray( (grid2d.u).data(), dev_grid2d.cuArr_u, 0,0, dev_grid2d.NFLAT()*sizeof(float2), cudaMemcpyDeviceToHost)

	checkCudaErrors(
		cudaMemcpyFromArray( dev_grid2d.dev_u, dev_grid2d.cuArr_u,
							0,0, sizeof(float2)*dev_grid2d.NFLAT(), cudaMemcpyDeviceToDevice) );
	float2 tempu[ dev_grid2d.NFLAT() ];
	
	checkCudaErrors(
		cudaMemcpy( tempu, dev_grid2d.dev_u, 
						sizeof(float2)*dev_grid2d.NFLAT(), cudaMemcpyDeviceToHost) );
*/


/*
	// C++ file Input/Output <fstream>
	std::ofstream addrxyf2_u_x_file, addrxyf2_u_y_file ;
	addrxyf2_u_x_file.open("./dataout/addrxyf2_u_x.csv");
	addrxyf2_u_y_file.open("./dataout/addrxyf2_u_y.csv");
	for (auto j=0; j<grid2d.Ld[1]; ++j) {
		// print first column's element
		addrxyf2_u_x_file   << grid2d.u[ 0 + j * grid2d.Ld[0] ].x ;
		addrxyf2_u_y_file   << grid2d.u[ 0 + j * grid2d.Ld[0] ].y ;
		
		// print remaining columns
		for (auto i=1; i<grid2d.Ld[0]; ++i) {
			addrxyf2_u_x_file  << ", " << grid2d.u[i+j*grid2d.Ld[0] ].x ;
			addrxyf2_u_y_file  << ", " << grid2d.u[i+j*grid2d.Ld[0] ].y ;
		}
		
		// print new line between rows
		addrxyf2_u_x_file  << std::endl;
		addrxyf2_u_y_file  << std::endl;

	}
	addrxyf2_u_x_file.close();
	addrxyf2_u_y_file.close();

	// copy results, output arrays from device to host memory				
	checkCudaErrors(
		cudaMemcpyFromArray( grid2d.u.data(), dev_grid2d.cuArr_u_out,
							0,0, dev_grid2d.NFLAT()*sizeof(float2), cudaMemcpyDeviceToHost) );
	// C++ file Input/Output <fstream>
	std::ofstream addrxyf2_u_out_x_file, addrxyf2_u_out_y_file ;
	addrxyf2_u_out_x_file.open("./dataout/addrxyf2_u_out_x.csv");
	addrxyf2_u_out_y_file.open("./dataout/addrxyf2_u_out_y.csv");
	for (auto j=0; j<grid2d.Ld[1]; ++j) {
		// print first column's element
		addrxyf2_u_out_x_file   << grid2d.u[ 0 + j * grid2d.Ld[0] ].x ;
		addrxyf2_u_out_y_file   << grid2d.u[ 0 + j * grid2d.Ld[0] ].y ;
		// print remaining columns
		for (auto i=1; i<grid2d.Ld[0]; ++i) {
			addrxyf2_u_out_x_file  << ", " << grid2d.u[i+j*grid2d.Ld[0] ].x ;
			addrxyf2_u_out_y_file  << ", " << grid2d.u[i+j*grid2d.Ld[0] ].y ;
		}
		// print new line between rows
		addrxyf2_u_out_x_file  << std::endl;
		addrxyf2_u_out_y_file  << std::endl;

	}
	addrxyf2_u_out_x_file.close();
	addrxyf2_u_out_y_file.close();
*/	
	
	
//	checkCudaErrors(
//		cudaFree( dev_grid2d.dev_f ) );
	
	
	
	return 0;
}
