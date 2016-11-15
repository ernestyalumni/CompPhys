/* simplelinear2dtex.cu
 * sine * sine function over 2-dimensional textured grid
 * Ernest Yeung  ernestyalumni@gmail.com
 * 20161113
 * 
 * Compilation tips if you're not using a make file
 * 
 * nvcc -std=c++11 -c ./physlib/R2grid.cpp -o R2grid.o  // or 
 * g++ -std=c++11 -c ./physlib/R2grid.cpp -o R2grid.o
 * 
 * nvcc -std=c++11 -c ./physlib/dev_R2grid.cu -o dev_R2grid.o
 * nvcc -std=c++11 simplelinear2dtex.cu R2grid.o dev_R2grid.o -o simplelinear2dtex
 * 
 */
#include <iostream> // std::cout
#include <fstream>  // std::ofstream  

#include "./physlib/R2grid.h"   // Grid2d
#include "./physlib/dev_R2grid.h"  // dev_Grid2d

constexpr const int WIDTH  { 16 } ;
constexpr const int HEIGHT { 8 } ;

__global__ void copyout(float* dev_out, 
						cudaTextureObject_t texObj,
						int width, int height)
{
	const int k_x = threadIdx.x + blockDim.x * blockIdx.x ; 
	const int k_y = threadIdx.y + blockDim.y * blockIdx.y ; 

	const int k   = k_x + k_y * width ; 
	
	if ( ( k_x >= width ) || ( k_y >= height ) ) {
		return ; }

	// Read from texture and write to global memory
	float c;
	c = tex2D<float>(texObj, k_x,k_y) ; 
	dev_out[k] = c ; 
}

__global__ void dkernel(float* dev_out, 
							cudaTextureObject_t texObj,
							const int width, const int height) {
	const int k_x = threadIdx.x + blockDim.x * blockIdx.x;
	const int k_y = threadIdx.y + blockDim.y * blockIdx.y;
	const int k   = k_x + width * k_y ; 

	if ( ( k_x >= width ) || ( k_y >= height ) ) {
		return ; }

	float c, r; 
	c = tex2D<float>(texObj, k_x, k_y);
	r = tex2D<float>(texObj, k_x+1, k_y);

	dev_out[k] = ( c + r  ) / (2.f); 
}

__global__ void addl_kernel(float* dev_out, 
							cudaTextureObject_t texObj,
							const int width, const int height) {
	const int k_x = threadIdx.x + blockDim.x * blockIdx.x;
	const int k_y = threadIdx.y + blockDim.y * blockIdx.y;
	const int k   = k_x + width * k_y ; 

	if ( ( k_x >= width ) || ( k_y >= height ) ) {
		return ; }

	float c, l; 
	c = tex2D<float>(texObj, k_x, k_y);
	l = tex2D<float>(texObj, k_x-1, k_y);

	dev_out[k] = ( c + l  ) / (0.2f); 
}

__global__ void dx_kernel(float* dev_out,
							cudaTextureObject_t texObj,
							const int width, const int height,
							float dx ) {
	const int k_x = threadIdx.x + blockDim.x * blockIdx.x;
	const int k_y = threadIdx.y + blockDim.y * blockIdx.y;
	const int k   = k_x + width * k_y;
	
	if ( ( k_x >= width ) || (k_y >= height ) ) {
		return ; }
		
	float l,r;
	r = tex2D<float>(texObj, k_x+1, k_y);
	l = tex2D<float>(texObj, k_x-1, k_y);
	
	dev_out[k] = (r-l) / (2.f * dx ) ;
}
	
__global__ void dy_kernel(float* dev_out,
							cudaTextureObject_t texObj,
							const int width, const int height,
							float dy ) {
	const int k_x = threadIdx.x + blockDim.x * blockIdx.x;
	const int k_y = threadIdx.y + blockDim.y * blockIdx.y;
	const int k   = k_x + width * k_y;
	
	if ( ( k_x >= width ) || (k_y >= height ) ) {
		return ; }
		
	float t,b;
	t = tex2D<float>(texObj, k_x, k_y+1);
	b = tex2D<float>(texObj, k_x, k_y-1);
	
	dev_out[k] = (t-b) / (2.f * dy ) ;
}



int main(int argc, char *argv[]) {
	constexpr const int DISPLAY_SIZE { 14 };

	const float PI { acos(-1.f) };
	
	constexpr std::array<int,2> LdS {WIDTH, HEIGHT };
	constexpr std::array<float,2> ldS {1.f, 1.f };

	Grid2d grid2d( LdS, ldS);
	
	// sanity check
	std::cout << "This is the value of pi : " << PI << std::endl;
	std::cout << "Size of rho on grid2d?  Do grid2d.rho.size() : " << grid2d.rho.size() << std::endl;
	std::cout << "Initially, on grid2d.rho : " << std::endl;
	for (auto i = 0; i < DISPLAY_SIZE; ++i ) {
		std::cout << " " << grid2d.rho[i] ; }
	std::cout << std::endl;

	for (auto j = 0; j < grid2d.Ld[1] ; ++j ) { 
		for (auto i = 0; i < grid2d.Ld[0] ; ++i ) {
			grid2d.rho[ grid2d.flatten(i,j) ] = static_cast<float>(grid2d.flatten(i,j) ) + 0.1f ; }
	}

	// sanity check
	std::cout << "grid2d.rho, after initializing with values given by flattened(x,y) + 0.1f : " << std::endl;
	for (auto i = 0; i < DISPLAY_SIZE; ++i ) {
		std::cout << " " << grid2d.rho[i + WIDTH*1] ; }
	std::cout << std::endl;

	cudaChannelFormatDesc channelDesc = 
		cudaCreateChannelDesc<float>();
		
	cudaArray* cuArray;
	cudaMallocArray(&cuArray, &channelDesc, WIDTH, HEIGHT); 
	
	// Copy to device memory some data located at address grid2d.rho 
	// in host memory
	cudaMemcpyToArray(cuArray, 0, 0, (grid2d.rho).data(), sizeof(float)*grid2d.NFLAT(), cudaMemcpyHostToDevice) ;
	
	// Specify texture
	struct cudaResourceDesc resDesc;
	memset(&resDesc,0,sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = cuArray;
	
	// Specify texture object parameters
	struct cudaTextureDesc texDesc;
	memset(&texDesc, 0,sizeof(texDesc));
//	texDesc.addressMode[0]  = cudaAddressModeClamp;
//	texDesc.addressMode[1]  = cudaAddressModeClamp;
	texDesc.addressMode[0]  = cudaAddressModeWrap;
	texDesc.addressMode[1]  = cudaAddressModeClamp;
//	texDesc.filterMode      = cudaFilterModeLinear;
	texDesc.filterMode      = cudaFilterModePoint;
	texDesc.readMode       = cudaReadModeElementType;
	
	// Create texture object
	cudaTextureObject_t texObj = 0;
	cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
	
	
	// Allocate result of transformation in device memory
	dim3 dev_L2 { static_cast<unsigned int>(WIDTH), 
					static_cast<unsigned int>(HEIGHT) };
	dev_Grid2d dev_grid2d( dev_L2) ;
	
	
	// Invoke kernel
	// MANUALLY CHANGE M_i here
	const dim3 M_i(4,2) ;
	const dim3 gridSize((WIDTH + M_i.x - 1)/M_i.x,
						(HEIGHT+ M_i.y - 1)/M_i.y);
	
	copyout<<<gridSize,M_i>>>(dev_grid2d.dev_rho_out, texObj, WIDTH, HEIGHT);
	
	
	// copy result, output array from device to host memory
	cudaMemcpy(grid2d.rho_out.data(), dev_grid2d.dev_rho_out, 
				sizeof(float)*dev_grid2d.NFLAT(), cudaMemcpyDeviceToHost);

	
	// sanity check: print out, read out results
	std::cout << "After kernel, which has a tex2D, and after cudaMemcpy, " << std::endl;
	for (int i = 0; i < DISPLAY_SIZE; ++i) {
		std::cout << " " << grid2d.rho_out[ i + WIDTH*1 ] ; }
	std::cout << std::endl;
	std::cout << "Size of rho_out on grid2d, grid2d.rho_out.size() : " << grid2d.rho_out.size() << std::endl;


	// C++ file Input/Output <fstream>
	// cf. http://stackoverflow.com/questions/25201131/writing-csv-files-from-c and C++ Primer, Lippman,Lajoie,..
	std::ofstream output_file;
	output_file.open("simplelinear2dtex_result.csv");
	for (auto j=0; j<grid2d.Ld[1]; ++j) {
		// print first column's element
		output_file << grid2d.rho_out[ 0 + j * grid2d.Ld[0] ] ;
		
		// print remaining columns
		for (auto i=1; i<grid2d.Ld[0]; ++i) {
			output_file << ", " << grid2d.rho_out[i+j*grid2d.Ld[0] ] ;
		}
	
		// print new line between rows
		output_file << std::endl;
	}
	
	std::ofstream ogref_file;
	ogref_file.open("simplelinear2dtex_ogref.csv"); // original reference
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
	
	output_file.close();
	ogref_file.close();
	
	// sanity check
	std::cout << " grid2d.hd[0] : " << grid2d.hd[0] << " grid2d.hd[1] : " << grid2d.hd[1] << std::endl;
	
	// Going further, 
	dkernel<<<gridSize,M_i>>>(dev_grid2d.dev_rho_out, texObj, WIDTH, HEIGHT ) ;

	// copy result, output array from device to host memory
	cudaMemcpy(grid2d.rho_out.data(), dev_grid2d.dev_rho_out, 
				sizeof(float)*dev_grid2d.NFLAT(), cudaMemcpyDeviceToHost);

	// sanity check: print out, read out results
	std::cout << "After dkernel, which has a tex2D, and after cudaMemcpy, " << std::endl;
	for (int i = 0; i < DISPLAY_SIZE; ++i) {
		std::cout << " " << grid2d.rho_out[i+1*WIDTH] ; }
	std::cout << std::endl;
	std::cout << "Size of rho_out on grid2d, grid2d.rho_out.size() : " << grid2d.rho_out.size() << std::endl;

	// C++ file Input/Output <fstream>
	std::ofstream doutput_file;
	doutput_file.open("simplelinear2dtex_d_result.csv");
	for (auto j=0; j<grid2d.Ld[1]; ++j) {
		// print first column's element
		doutput_file << grid2d.rho_out[ 0 + j * grid2d.Ld[0] ] ;
		
		// print remaining columns
		for (auto i=1; i<grid2d.Ld[0]; ++i) {
			doutput_file << ", " << grid2d.rho_out[i+j*grid2d.Ld[0] ] ;
		}
		
		// print new line between rows
		doutput_file << std::endl;
	}
	doutput_file.close();

	// Going further, with addl_kernel
	addl_kernel<<<gridSize,M_i>>>(dev_grid2d.dev_rho_out, texObj, WIDTH, HEIGHT ) ;

	// copy result, output array from device to host memory
	cudaMemcpy(grid2d.rho_out.data(), dev_grid2d.dev_rho_out, 
				sizeof(float)*dev_grid2d.NFLAT(), cudaMemcpyDeviceToHost);

	// sanity check: print out, read out results
	std::cout << "After addl_kernel, which has a tex2D, and after cudaMemcpy, " << std::endl;
	for (int i = 0; i < DISPLAY_SIZE; ++i) {
		std::cout << " " << grid2d.rho_out[i+1*WIDTH] ; }
	std::cout << std::endl;
	std::cout << "Size of rho_out on grid2d, grid2d.rho_out.size() : " << grid2d.rho_out.size() << std::endl;

	// C++ file Input/Output <fstream>
	std::ofstream addl_output_file;
	addl_output_file.open("simplelinear2dtex_add_l_result.csv");
	for (auto j=0; j<grid2d.Ld[1]; ++j) {
		// print first column's element
		addl_output_file << grid2d.rho_out[ 0 + j * grid2d.Ld[0] ] ;
		
		// print remaining columns
		for (auto i=1; i<grid2d.Ld[0]; ++i) {
			addl_output_file << ", " << grid2d.rho_out[i+j*grid2d.Ld[0] ] ;
		}
		
		// print new line between rows
		addl_output_file << std::endl;
	}
	addl_output_file.close();

	// sanity check
	std::cout << " grid2d.hd[0] : " << grid2d.hd[0] << " grid2d.hd[1] : " << grid2d.hd[1] << std::endl;

	// Going further, with dx_kernel
	dx_kernel<<<gridSize,M_i>>>(dev_grid2d.dev_rho_out, texObj, WIDTH, HEIGHT, grid2d.hd[0] ) ;

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
	dx_output_file.open("simplelinear2dtex_dx_result.csv");
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

	// Going further, with dy_kernel
	dy_kernel<<<gridSize,M_i>>>(dev_grid2d.dev_rho_out, texObj, WIDTH, HEIGHT, grid2d.hd[1] ) ;

	// copy result, output array from device to host memory
	cudaMemcpy(grid2d.rho_out.data(), dev_grid2d.dev_rho_out, 
				sizeof(float)*dev_grid2d.NFLAT(), cudaMemcpyDeviceToHost);

	// sanity check: print out, read out results
	std::cout << "After dy_kernel, which has a tex2D, and after cudaMemcpy, " << std::endl;
	for (int i = 0; i < DISPLAY_SIZE; ++i) {
		std::cout << " " << grid2d.rho_out[i+1*WIDTH] ; }
	std::cout << std::endl;
	std::cout << "Size of rho_out on grid2d, grid2d.rho_out.size() : " << grid2d.rho_out.size() << std::endl;

	// C++ file Input/Output <fstream>
	std::ofstream dy_output_file;
	dy_output_file.open("simplelinear2dtex_dy_result.csv");
	for (auto j=0; j<grid2d.Ld[1]; ++j) {
		// print first column's element
		dy_output_file << grid2d.rho_out[ 0 + j * grid2d.Ld[0] ] ;
		
		// print remaining columns
		for (auto i=1; i<grid2d.Ld[0]; ++i) {
			dy_output_file << ", " << grid2d.rho_out[i+j*grid2d.Ld[0] ] ;
		}
		
		// print new line between rows
		dy_output_file << std::endl;
	}
	dy_output_file.close();

	

	// Destroy texture object
	cudaDestroyTextureObject(texObj);
	
	// Free device memory
	cudaFreeArray(cuArray);

	
	return 0;
}
