/* sinsin2dtex.cu
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
 * nvcc -std=c++11 sinsin2dtex.cu R2grid.o dev_R2grid.o -o sinsins2tex_cu
 * 
 */
#include <iostream> // std::cout
#include <fstream>  // std::ofstream  

#include "./physlib/R2grid.h"   // Grid2d
#include "./physlib/dev_R2grid.h"  // dev_Grid2d

constexpr const int WIDTH  { 640 } ;
constexpr const int HEIGHT { 640 } ;

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
	dev_out[k] = tex2D<float>(texObj, k_x,k_y);
}

int main(int argc, char *argv[]) {
	constexpr const int DISPLAY_SIZE { 22 };

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

	std::array<int,2> ix_in { 0, 0 };
	std::array<float,2> Xi { 0.f, 0.f };
	for (auto j = 0; j < grid2d.Ld[0] ; ++j ) { 
		for (auto i = 0; i < grid2d.Ld[1] ; ++i ) {
			ix_in[0] = i ;
			ix_in[1] = j ;
			Xi = grid2d.gridpt_to_space( ix_in );	
			grid2d.rho[ grid2d.flatten(i,j) ] = sin( 2.f*PI*Xi[0])*sin(2.f*PI*Xi[1]) ; 
		}
	}

	// sanity check
	std::cout << "grid2d.rho, after initializing with values given by sin(2*pi*x)*sin(2*pi*y) : " << std::endl;
	for (auto i = 0; i < DISPLAY_SIZE; ++i ) {
		std::cout << " " << grid2d.rho[(i+WIDTH/4)+ HEIGHT/4*WIDTH] ; }
	std::cout << std::endl;

	// Allocate CUDA array in device memory
/*	cudaChannelFormatDesc channelDesc = 
		cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
*/
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
	texDesc.addressMode[0]  = cudaAddressModeClamp;
	texDesc.addressMode[1]  = cudaAddressModeClamp;
	texDesc.filterMode      = cudaFilterModeLinear;
	texDesc.readMode       = cudaReadModeElementType;
	
	// Create texture object
	cudaTextureObject_t texObj = 0;
	cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
	
	
	// Allocate result of transformation in device memory
	dim3 dev_L2 { static_cast<unsigned int>(WIDTH), 
					static_cast<unsigned int>(HEIGHT) };
	dev_Grid2d dev_grid2d( dev_L2) ;
	
	
	// Invoke kernel
	const dim3 M_i(16,16) ;
	const dim3 gridSize((WIDTH + M_i.x - 1)/M_i.x,
						(HEIGHT+ M_i.y - 1)/M_i.y);
	
	copyout<<<gridSize,M_i>>>(dev_grid2d.dev_rho_out, texObj, WIDTH, HEIGHT);
	
	
	// copy result, output array from device to host memory
	cudaMemcpy(grid2d.rho_out.data(), dev_grid2d.dev_rho_out, 
				sizeof(float)*dev_grid2d.NFLAT(), cudaMemcpyDeviceToHost);
	
	// sanity check: print out, read out results
	std::cout << "After kernel, which has a tex2D, and after cudaMemcpy, " << std::endl;
	for (int i = 0; i < DISPLAY_SIZE; ++i) {
		std::cout << " " << grid2d.rho_out[(i+WIDTH/4)+HEIGHT/4*WIDTH] ; }
	std::cout << std::endl;
	std::cout << "Size of rho_out on grid2d, grid2d.rho_out.size() : " << grid2d.rho_out.size() << std::endl;

	
	// Destroy texture object
	cudaDestroyTextureObject(texObj);
	
	// Free device memory
	cudaFreeArray(cuArray);
	
	
	// C++ file Input/Output <fstream>
	// cf. http://stackoverflow.com/questions/25201131/writing-csv-files-from-c and C++ Primer, Lippman,Lajoie,..
	std::ofstream output_file;
	output_file.open("sinsin2dtex_result.csv");
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
	ogref_file.open("sinsin2dtex_ogref.csv"); // original reference
	for (auto j=0; j<grid2d.Ld[1]; ++j) {
		// print first column's element
		ogref_file << grid2d.rho_out[ 0 + j * grid2d.Ld[0] ] ;
		
		// print remaining columns
		for (auto i=1; i<grid2d.Ld[0]; ++i) {
			ogref_file << ", " << grid2d.rho_out[i+j*grid2d.Ld[0] ] ;
		}
		
		// print new line between rows
		ogref_file << std::endl;
	}
	
	output_file.close();
	ogref_file.close();
	
	
	return 0;
}
