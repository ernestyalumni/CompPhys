/* texdynamics2d.cu
 * texture dynamics, in 2-dimensions
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
#include "./commonlib/checkerror.h" // checkCudaErrors

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

__global__ void addr_kernel(float* dev_out,
							cudaTextureObject_t texObj,
							const int width, const int height) {
	const int k_x = threadIdx.x + blockDim.x * blockIdx.x;
	const int k_y = threadIdx.y + blockDim.y * blockIdx.y;
	const int k   = k_x + width * k_y;
	
	if ((k_x >= width) || (k_y >= height) ){
		return ; }
	
	float c,r;
	c = tex2D<float>(texObj,k_x,k_y) ;
	r = tex2D<float>(texObj,k_x+1,k_y);
	
	dev_out[k] = c+r;
}

__global__ void addl_kernel(float* dev_out,
							cudaTextureObject_t texObj,
							const int width, const int height) {
	const int k_x = threadIdx.x + blockDim.x * blockIdx.x;
	const int k_y = threadIdx.y + blockDim.y * blockIdx.y;
	const int k   = k_x + width * k_y;
	
	if ((k_x >= width) || (k_y >= height) ){
		return ; }
	
	float c,l;
	c = tex2D<float>(texObj,k_x,k_y) ;
	l = tex2D<float>(texObj,k_x-1,k_y);
	
	dev_out[k] = c+l;
}


void dxkernel_launcher(float* f_in, float* f_out, cudaTextureObject_t& texfin, cudaTextureObject_t& texfout,
						const int M_x, const int M_y, 
						const int WIDTH, const int HEIGHT, const float DX, 
						const int NITERS=1) {
	const dim3 M_i(M_x,M_y) ;
	const dim3 gridSize((WIDTH  + M_i.x - 1)/M_i.x,
						(HEIGHT + M_i.y - 1)/M_i.y);
	
	volatile bool dstOut = true;
	
	for (auto iter = 0; iter < NITERS; ++iter) {
//		float *in, *out;
//		cudaTextureObject_t texin, texout ;
		float *out;
		cudaTextureObject_t texin ;

/*		if (dstOut) {
			in     = f_in;
			out    = f_out; 
			texin  = texfin;
			texout = texfout; }
		else {
			out    = f_in;
			in     = f_out; 
			texout = texfin;
			texin  = texfout; 
		} */

		if (dstOut) {
			out   = f_out;
			texin = texfin; }
		else {
			out   = f_in;
			texin = texfout; }
				

		dx_kernel<<<gridSize,M_i>>>( out, texin, WIDTH, HEIGHT, DX );
			
		dstOut = !dstOut;
	}
}

void add_kernel(float *dev_out, 
				cudaTextureObject_t texObj,
				const int M_x, const int M_y,
				const int L_x, const int L_y,
				const int NITERS=1) {
	const dim3 M_i(M_x,M_y) ;
	const dim3 gridSize((L_x+M_i.x-1)/M_i.x, 
						(L_y+M_i.y-1)/M_i.y) ;
	
	for (auto iter = 0; iter < NITERS; ++iter) {
		addr_kernel<<<gridSize,M_i>>>(dev_out, texObj, L_x,L_y) ;
	}
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
	
	/*
	// Create texture object
	TexObj2d tex_f_in( dev_grid2d.dev_f_in, dev_grid2d.NFLAT() ) ;
	TexObj2d tex_f_out( dev_grid2d.dev_f_out, dev_grid2d.NFLAT() ) ;
	TexObj2d tex_f_constscr( dev_grid2d.dev_f_constsrc, dev_grid2d.NFLAT() ) ;
*/

	// initial values
//	std::array<int,2> ix_in { 0,0};
//	std::array<float,2> Xi {0.f, 0.f };
	for (auto j = 0; j < grid2d.Ld[1] ; ++j) { 
		for (auto i = 0; i < grid2d.Ld[0]; ++i ) {
//			ix_in[0] = i;
//			ix_in[1] = j;
//			Xi = grid2d.gridpt_to_space( ix_in );
			// for sin*sin
//			grid2d.rho[ grid2d.flatten(i,j) ] = sin( Xi[0]) * sin( Xi[1] ) ; 
			// for linear
			grid2d.rho[ grid2d.flatten(i,j) ] = static_cast<float>( grid2d.flatten(i,j) ) + 0.1f ; 
		}
	}
	
	// sanity check
	std::cout << "grid2d.rho, after initializing with values given by linear : " << std::endl;
	for (auto i = 0; i < DISPLAY_SIZE; ++i ) {
		std::cout << " " << grid2d.rho[(i+WIDTH/4)+ HEIGHT/4*WIDTH] ; }
	std::cout << std::endl;
	
	// Copy to device memory some data located at address grid2d.rho in host memory
	checkCudaErrors( cudaMemcpy(dev_grid2d.dev_f_in, 
						(grid2d.rho).data(), sizeof(float)*grid2d.NFLAT(), cudaMemcpyHostToDevice) );

	// Create texture object
	TexObj2d tex_f_in( dev_grid2d.dev_f_in, dev_grid2d.NFLAT() ) ;
	TexObj2d tex_f_out( dev_grid2d.dev_f_out, dev_grid2d.NFLAT() ) ;
	TexObj2d tex_f_constscr( dev_grid2d.dev_f_constsrc, dev_grid2d.NFLAT() ) ;


	// Invoke kernel
	// MANUALLY CHANGE M_i here
	constexpr const int M_X { 8 }; 
	constexpr const int M_Y { 4 };
	// sanity check on single launch
	

//	add_kernel( dev_grid2d.dev_f_out, tex_f_in.texObj, M_X, M_Y, WIDTH, HEIGHT );
	const dim3 M_i(M_X,M_Y) ;
	const dim3 gridSize((WIDTH + M_i.x - 1)/M_i.x,
						(HEIGHT + M_i.y - 1)/M_i.y);
	
	addr_kernel<<<gridSize,M_i>>>(dev_grid2d.dev_f_out, tex_f_in.texObj, WIDTH, HEIGHT) ;
							
	
	// C++ file Input/Output <fstream>
	float og_printout[grid2d.NFLAT()];
	checkCudaErrors(
		cudaMemcpy( og_printout, dev_grid2d.dev_f_in, sizeof(float)*dev_grid2d.NFLAT(), cudaMemcpyDeviceToHost) 
	);
	std::ofstream ogref_file;
	ogref_file.open("./dataout/ogref_linear.csv"); // original reference
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
	ogref_file.close();

	// copy result, output array from device to host memory
	cudaMemcpy(grid2d.rho_out.data(), dev_grid2d.dev_f_out, 
				sizeof(float)*dev_grid2d.NFLAT(), cudaMemcpyDeviceToHost);

	// C++ file Input/Output <fstream>
	std::ofstream addr01_file;
	addr01_file.open("./dataout/addr01_result.csv");
	for (auto j=0; j<grid2d.Ld[1]; ++j) {
		// print first column's element
		addr01_file << grid2d.rho_out[ 0 + j * grid2d.Ld[0] ] ;
		
		// print remaining columns
		for (auto i=1; i<grid2d.Ld[0]; ++i) {
			addr01_file << ", " << grid2d.rho_out[i+j*grid2d.Ld[0] ] ;
		}
		
		// print new line between rows
		addr01_file << std::endl;
	}
	addr01_file.close();


	return 0;
}
