/* surfObj2d.h
 * surface memory Object class, 2-dim. 
 * Ernest Yeung  ernestyalumni@gmail.com
 * 20161116
 * 
 * cf. http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#surface-object-api
 * 3.2.11.2.1. Surface Object API
 * 
 * compilation tip: (compile separately)
 * nvcc -std=c++11 -c ./commonlib/surfObj2d.cu -o surfObj2d.o
 * 
 */
#ifndef __SURFOBJ2D_H__
#define __SURFOBJ2D_H__

#include "checkerror.h" // checkCudaErrors

class SurfObj2d
{
	public :
		// Specify surface
//		struct cudaResourceDesc resDesc;  // old-school C style
		cudaResourceDesc resDesc;
		
		// Specify surface object parameters
//		struct cudaTextureDesc texDesc; // old-school C style
		cudaTextureDesc texDesc;

		// Create surface object
		cudaSurfaceObject_t surfObj;

		
		// constructor
		__host__ SurfObj2d( cudaArray*  ) ;

		// constructor
		// Note: you're going to have to set the cudaArray this texture object is associated to at a later time.
		__host__ SurfObj2d(  ) ;

		
		// destructor
		__host__ ~SurfObj2d(); 
		
};

#endif // __SURFOBJ2D_H__

