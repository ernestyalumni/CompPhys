/*
 * gaussianprof.cu
 * gaussian profile on the GPU
 * Ernest Yeung ernestyalumni@gmail.com
 * 20160617
 * 
 * Compilation: I did this and it worked
 * nvcc gaussianprof.cu -o gaussianprof -lglut -lGL
 * 
 * */
//#include <iostream>
//using namespace std;
#include "./common/gpu_bitmap.h"
#include "math.h" // cf. https://developer.nvidia.com/cuda-math-library CUDA Math Library

// Constants that changes "scaling", what pts. are included in Julia set, and such
#define NCELLS 800
#define DIMY 800

//const double RHO0   = 0.656; // 0.656 kg/m^3 density at 25C, 1 atm
//const double L_0    = 1.0;
//const double DELTAx =  L_0/((double) NCELLS);

#define RHO0 0.656
#define L_0 1.0
#define DELTAx L_0/((double) NCELLS)

__device__ double gaussian( double xvar, double A, double k, double x_0 )
{
	return A*exp(-k*(xvar-x_0)*(xvar-x_0) );
}

	
__global__ void kernel( uchar4 *ptr ) {
  // map from threadIdx/BlockIdx to pixel position
	int x = threadIdx.x + blockIdx.x * blockDim.x ;
//	int offset = x+y*gridDim.x * blockDim.x;

	// now calculate the value at that position
	double value = gaussian( ((double) x)*DELTAx, RHO0, 1./sqrt(0.00001), 0.25 ); 

	// calculate how many blocks in y-direction to fill up
	int ivalue = ((int) (value*DIMY)) ;
	
	// remember that ptr is a pointer to uchar4 that CUDA and OpenGL shares to render bitmaps
/*
	for (int j = 0; j<ivalue ; ++j) {
		int offset = x + j * gridDim.x * blockDim.x ;
		ptr[offset].x = 0;
		ptr[offset].y = 255;
		ptr[offset].z = 0;
		ptr[offset].w = 255;
	} 
	for (int j = ivalue; j<DIMY; ++j) {
		int offset = x + j * gridDim.x * blockDim.x ;
		ptr[offset].x = 255;
		ptr[offset].y = 0;
		ptr[offset].z = 0;
		ptr[offset].w = 255;
	}
	* */
	
	
	for (int j = 0; j<DIMY; ++j) {
		int offset = x + j * gridDim.x * blockDim.x ;
		if (j < ivalue) {
			ptr[offset].x = 0;
			ptr[offset].y = 255;
			ptr[offset].z = 0;
			ptr[offset].w = 255;
		} else {
			ptr[offset].x = 255;
			ptr[offset].y = 0;
			ptr[offset].z = 0;
			ptr[offset].w = 255;
		}
	}		
	
/*
	// test case
	for (int j = 0; j<DIMY; ++j) {
		int offset = x + j * gridDim.x * blockDim.x ;
		if (j < DIMY/2) {
			ptr[offset].x = 0;
			ptr[offset].y = 255;
			ptr[offset].z = 0;
			ptr[offset].w = 255;
		} else {
			ptr[offset].x = 255;
			ptr[offset].y = 0;
			ptr[offset].z = 0;
			ptr[offset].w = 255;
		}
	}		
*/
} 

int main(int argc, char *argv[]) {
	GPUBitmap bitmap( NCELLS, DIMY );
	
	int Mthreads = 40;	
	kernel<<<NCELLS/Mthreads,Mthreads>>>(bitmap.devPtr);

	bitmap.display_and_exit();
}

