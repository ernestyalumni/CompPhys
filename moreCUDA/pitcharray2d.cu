/*
 * pitcharray2d.cu
 * 2-dimensional array allocation on device with pitch, pitched pointer
 * Ernest Yeung  ernestyalumni@gmail.com
 * cf. Steven had an excellent blog post:
 * http://www.stevenmarkford.com/allocating-2d-arrays-in-cuda/
 * 20160706
*/
// kernel which copies data from d_array to destinationArray
#include <iostream>

__global__ void CopyData(float* d_array, float* destinationArray, size_t pitch, 
	int N_i, int N_j) // N_i is the total number of rows; N_j is total number of columns
{
	for (int i = 0; i < N_i; ++i)
	{
		// update the pointer to point to the beginning of the next row
		float* row = (float*)(((char*) d_array) + (i * pitch));
		
		for (int j = 0; j < N_j; ++j)
		{
			row[j] = 123.0 ; // make every value in the array 123.0
			destinationArray[(i*N_j) + j] = row[j];
		}
	}
}

int main(int argc, char** argv)
{
	int N_j { 15 };
	int N_i { 10 };
	float* d_array; // the device array which memory will be allocated to
	float* d_destinationArray; // device array
	// allocate memory on the host
	float* h_array = new float[N_j*N_i];
	
	// the pitch value assigned by cudaMallocPitch
	// (which ensures correct data structure alignment)
	size_t pitch;
	
	// allocated the device memory for source array
	cudaMallocPitch(&d_array, &pitch, N_j*sizeof(float), N_i);
	
	// allocate the device memory for destination array
	cudaMalloc(&d_destinationArray, N_i*N_j*sizeof(float));
	
	// call the kernel which copies values from d_array to d_destinationArray
	CopyData<<<100,512>>>(d_array, d_destinationArray, pitch, N_i, N_j);
	
	// copy the data back to the host memory
	cudaMemcpy(h_array, 
						d_destinationArray,
						N_i*N_j*sizeof(float),
						cudaMemcpyDeviceToHost);
						
	// print out the values (all the values are 123.0)
	for (int i = 0 ; i < N_i; ++i)
	{
		for (int j = 0 ; j < N_j; ++j)
		{
			std::cout << "h_array[" << (i*N_j) +j << "]=" << h_array[(i*N_j)+j] << std::endl;
		}
	}
}
