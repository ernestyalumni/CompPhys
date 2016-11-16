/* simpletexdynamics2d.cu
 * simple texture dynamics, in 2-dimensions
 * Ernest Yeung  ernestyalumni@gmail.com
 * 20161115
 * 
 * Compilation tips if you're not using a make file
 * 
 * nvcc -std=c++11 -c ./physlib/R2grid.cpp -o R2grid.o  // or 
 * g++ -std=c++11 -c ./physlib/R2grid.cpp -o R2grid.o
 * 
 * nvcc -std=c++11 -c ./physlib/dev_R2grid.cu -o dev_R2grid.o
 * nvcc -std=c++11 -c ./commonlib/texObjCls2d.cu -o texObjCls2d.o
 * nvcc -std=c++11 simpletexdynamics2d.cu R2grid.o dev_R2grid.o texObjCls2d.o -o simpletexturedynamics2d
 * 
 */
#include <iostream> // std::cout
#include <fstream>  // std::ofstream

#include "./physlib/R2grid.h" // Grid2d
#include "./physlib/dev_R2grid.h" // dev_Grid2d
#include "./commonlib/texObjCls2d.h" // TexObj2d

constexpr const int WIDTH  { 128 } ;
constexpr const int HEIGHT { 64 } ;

__global__ void dx_kernel(float* dev_out, 
							cudaTextureObject_t texObj,
							const int width, const int height, 
							float dx ) {
	const int k_x = threadIdx.x + blockDim.x * blockIdx.x; 
	const int k_y = threadIdx.y + blockDim.y * blockIdx.y;
	const int k   = k_x + width * k_y;
	
	if ( ( k_x >= width ) || (k_y >= height) ) {
		return ; }
		
	float l,r ; 
	l = tex2D<float>(texObj,k_x-1,k_y);
	r = tex2D<float>(texObj,k_x+1,k_y);
	
	dev_out[k] = ( r-l)/(2.f * dx) ;
}


int main(int argc, char* argv[]) {
	constexpr const int DISPLAY_SIZE { 14 };
	
//	const float PI {acos(-1.f) };
	const float PI { 3.14159265358979323846 };
	
	constexpr std::array<int,2> LdS { WIDTH, HEIGHT };
	constexpr std::array<float,2> ldS { 2.f * PI , 2.f * PI };
	
	Grid2d grid2d{ LdS, ldS };
	
	// sanity check
	std::cout << "This is the value of pi : " << PI << std::endl;
	std::cout << "Size of rho on grid2d?  Do grid2d.rho.size() : " << grid2d.rho.size() << std::endl;
	std::cout << "Initially, on grid2d.rho : " << std::endl;
	for (auto i = 0; i < DISPLAY_SIZE; ++i ) {
		std::cout << " " << grid2d.rho[i] ; }
	std::cout << std::endl;
	// END of sanity check

	// Allocate device memory
	dim3 dev_L2 { static_cast<unsigned int>(WIDTH),
					static_cast<unsigned int>(HEIGHT) };
	dev_Grid2d dev_grid2d( dev_L2 );
	
	// Create texture object
	TexObj2d texObj2d( dev_grid2d.cuArr_rho ) ;

	// initial values
	std::array<int,2> ix_in { 0,0};
	std::array<float,2> Xi {0.f, 0.f };
	for (auto j = 0; j < grid2d.Ld[1] ; ++j) { 
		for (auto i = 0; i < grid2d.Ld[0]; ++i ) {
			ix_in[0] = i;
			ix_in[1] = j;
			Xi = grid2d.gridpt_to_space( ix_in );
			grid2d.rho[ grid2d.flatten(i,j) ] = sin( Xi[0]) * sin( Xi[1] ) ; 
		}
	}
	
	// sanity check
	std::cout << "grid2d.rho, after initializing with values given by sin(x)*sin(y) : " << std::endl;
	for (auto i = 0; i < DISPLAY_SIZE; ++i ) {
		std::cout << " " << grid2d.rho[(i+WIDTH/4)+ HEIGHT/4*WIDTH] ; }
	std::cout << std::endl;
	
	// Copy to device memory some data located at address grid2d.rho in host memory
	cudaMemcpyToArray(dev_grid2d.cuArr_rho, 0, 0, 
						(grid2d.rho).data(), sizeof(float)*grid2d.NFLAT(), cudaMemcpyHostToDevice) ;

	
	// Invoke kernel
	// MANUALLY CHANGE M_i here
	const dim3 M_i(8,4) ;
	const dim3 gridSize((WIDTH  + M_i.x - 1)/M_i.x,
						(HEIGHT + M_i.y - 1)/M_i.y);
						 
	dx_kernel<<<gridSize, M_i>>>( dev_grid2d.dev_rho_out, texObj2d.texObj, WIDTH, HEIGHT, grid2d.hd[0]);					 
						 
	// copy result, output array from device to host memory
	cudaMemcpy(grid2d.rho_out.data(), dev_grid2d.dev_rho_out, 
				sizeof(float)*dev_grid2d.NFLAT(), cudaMemcpyDeviceToHost);

	// sanity check: print out, read out results
	std::cout << "After dx_kernel, which has a tex2D, and after cudaMemcpy, " << std::endl;
	for (int i = 0; i < DISPLAY_SIZE; ++i) {
		std::cout << " " << grid2d.rho_out[i+1*WIDTH] ; }
	std::cout << std::endl;
	std::cout << "Size of rho_out on grid2d, grid2d.rho_out.size() : " << grid2d.rho_out.size() << std::endl;

	// C++ file Input/Output <fstream>
	std::ofstream dx_output_file;
	dx_output_file.open("simpletexdyn_dx_result.csv");
	for (auto j=0; j<grid2d.Ld[1]; ++j) {
		// print first column's element
		dx_output_file << grid2d.rho_out[ 0 + j * grid2d.Ld[0] ] ;
		
		// print remaining columns
		for (auto i=1; i<grid2d.Ld[0]; ++i) {
			dx_output_file << ", " << grid2d.rho_out[i+j*grid2d.Ld[0] ] ;
		}
		
		// print new line between rows
		dx_output_file << std::endl;
	}
	dx_output_file.close();				
	
	std::ofstream ogref_file;
	ogref_file.open("simpletexdyn_ogref.csv"); // original reference
	for (auto j=0; j<grid2d.Ld[1]; ++j) {
		// print first column's element
		ogref_file << grid2d.rho[ 0 + j * grid2d.Ld[0] ] ;
		
		// print remaining columns
		for (auto i=1; i<grid2d.Ld[0]; ++i) {
			ogref_file << ", " << grid2d.rho[i+j*grid2d.Ld[0] ] ;
		}
		
		// print new line between rows
		ogref_file << std::endl;
	}
	
	
	return 0;
}
