/*
 * gpujuliad.cu
 * make Julia sets on the GPU
 * Ernest Yeung ernestyalumni@gmail.com
 * 20160617
 * */
#include <iostream>
using namespace std;
#include <cstdlib> /* atof */ 
#include "./common/gpu_bitmap.h"

// Constants that changes "scaling", what pts. are included in Julia set, and such
#define DIM 1600
#define MAG_THR 1000 // magnitude threshold that determines if pt. is in Julia set
#define TESTITERS 500 // originally 200, tests further what points go to infinity; higher no. makes it "lacy"

// Constants that change formula for f, f(z) = z*z + c
#define CREAL -0.8168 // originally -0.8
#define CIMAG 0.14583 // originally 0.154

struct cuComplex {
  float r;
  float i;
  __device__ cuComplex( float a, float b) : r(a), i(b) {}
  __device__ float magnitude2( void ) {
    return r * r + i * i;
  }
  __device__ cuComplex operator*(const cuComplex& a) {
    return cuComplex(r*a.r - i*a.i, i*a.r + r*a.i);
  }
  __device__ cuComplex operator+(const cuComplex& a) {
    return cuComplex(r+a.r, i+a.i);
  }
};

__device__ int julia( int x, int y, const float scale) {
//  float jx = scale * (float)(DIM/2 - x)/(DIM/2);
//  float jy = scale * (float)(DIM/2 - y)/(DIM/2);
	float jx = scale * (float)(x - DIM/2)/(DIM/2);
	float jy = scale * (float)(y - DIM/2)/(DIM/2);

	cuComplex c(CREAL,CIMAG);
	cuComplex a(jx,jy);

	int i = 0;
	for (i=0; i<TESTITERS; i++) {
		a = a*a + c;
		if (a.magnitude2() > MAG_THR)
			return 0; // return 0 if it is not in set
	}
	return 1; // return 1 if point is in set
}

__global__ void kernel( uchar4 *ptr, const float scale) {
  // map from threadIdx/BlockIdx to pixel position
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int offset = x+y*gridDim.x * blockDim.x;

  // now calculate the value at that position
  int juliaValue = julia(x,y,scale);
  ptr[offset].x = 255 * juliaValue;  // red if julia() returns 1, black if pt. not in set
  ptr[offset].y = 0;
  ptr[offset].z = 0;
  ptr[offset].w = 255;
} 

int main(int argc, char *argv[]) {
	float scale;
	if (argc <= 1) { 
		scale = 1.5;
	}
	else {
		scale = (float) atof( argv[1] );
	}
	GPUBitmap bitmap( DIM, DIM );
	
	dim3 grid(DIM/4,DIM/4);
	dim3 block(4,4);
	
	// sanity check 
	cout << "This is the value for scale currently: " << scale << endl;
	
	kernel<<<grid,block>>>(bitmap.devPtr,scale);

	bitmap.display_and_exit();
}

