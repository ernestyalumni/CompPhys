/*
 * dev3darray.cu
 * 3-dimensional array allocation on device with cudaMalloc3DArray, demonstration
 * Ernest Yeung  ernestyalumni@gmail.com
 * 20160706
*/
#include <iostream> // cout
#include<limits> // // numeric_limits

__global__ void kernel(cudaArray* dev_array) {
	
}

int main(int argc, char *argv[]) {
	std::cout << " Size of float : " << sizeof(float) << std::endl;
	std::cout << " Size of char  : " << sizeof(char)  << std::endl;
	std::cout << " Size of char (in bits) : " << CHAR_BIT * sizeof(char) << std::endl;

	const int min_int { std::numeric_limits<int>::min() };
	const int max_int { std::numeric_limits<int>::max() };
	std::cout << " minimum range for int : " << min_int << std::endl;
	std::cout << " maximum range for int : " << max_int << std::endl;
	
	const int min_float { std::numeric_limits<float>::min() };
	const int max_float { std::numeric_limits<float>::max() };
	std::cout << " minimum range for float : " << min_float << std::endl;
	std::cout << " maximum range for float : " << max_float << std::endl;



	const int N_x { 300 }, N_y { 300 }, N_z { 300 };

	cudaExtent extent = make_cudaExtent( N_x*sizeof(float), N_y, N_z);
//	cudaExtent extent = make_cudaExtent( N_x, N_y, N_z)
	
	cudaChannelFormatDesc channeldesc { sizeof(float)*CHAR_BIT, 0, 0, 0, cudaChannelFormatKindFloat};
	cudaArray* dev_array;
	cudaMalloc3DArray(&dev_array, &channeldesc, extent);
	
	cudaFreeArray(dev_array);

	cudaDeviceReset();
}
