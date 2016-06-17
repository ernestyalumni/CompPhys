/*
 * graphicsinteropb.cu
 * make OpenGL bitmaps on the GPU
 * Ernest Yeung ernestyalumni@gmail.com
 * 20160616
 * */
#define DIM 1024

#include "./common/gpu_bitmap.h"
#include "./common/errors.h"

#include <cmath> 
  
// based on ripple code, but uses uchar4, which is the 
// type of data graphic interop uses
__global__ void kernel( uchar4 *ptr ) {
	// map from threadIdx/BlockIdx to pixel position
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;
	
	// now calculate the value at that position
	float fx = x - DIM/2.;
	float fy = y - DIM/2.;
	unsigned char green = 128 + 127 * sin( abs(fx*100) - abs(fy*100) );
	
	// accessing uchar4 vs. unsigned char*
	ptr[offset].x = 0 ;
	ptr[offset].y = green;
	ptr[offset].z = 0;
	ptr[offset].w = 255;
/* important thing to realize is that this image will be handed directly to OpenGL */
}


int main( int argc, char **argv) {
	GPUBitmap bitmap(DIM, DIM );							
	
	dim3 grids(DIM/16,DIM/16);
	dim3 threads(16,16);
	kernel<<<grids,threads>>>( bitmap.devPtr );
		
	bitmap.display_and_exit();

	
};
												
	
	
