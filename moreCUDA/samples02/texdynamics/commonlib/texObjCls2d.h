/* texObjCls2d.h
 * texture memory Object class, 2-dim. 
 * Ernest Yeung  ernestyalumni@gmail.com
 * 20161116
 * 
 * cf. https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-kepler-texture-objects-improve-performance-and-flexibility/
 * M. Clark, CUDA Pro Tip: Kepler Texture Objects Improve Performance and Flexibility
 * 
 * compilation tip: (compile separately)
 * nvcc -std=c++11 -c ./commonlib/texObjCls2d.cu -o texObjCls2d.o
 * 
 */
#ifndef __TEXOBJCLS2D_H__
#define __TEXOBJCLS2D_H__

#include "checkerror.h" // checkCudaErrors

class TexObj2d
{
	public :
		// Specify texture
//		struct cudaResourceDesc resDesc;  // old-school C style
		cudaResourceDesc resDesc;
		
		// Specify texture object parameters
//		struct cudaTextureDesc texDesc; // old-school C style
		cudaTextureDesc texDesc;

		// Create texture object
		cudaTextureObject_t texObj;

		
		// constructor
		__host__ TexObj2d( float*, const int ) ;

		// constructor
		// Note: you're going to have to set the cudaArray this texture object is associated to at a later time.
		__host__ TexObj2d(  ) ;

		
		// destructor
		__host__ ~TexObj2d(); 
		
};

#endif // __DEV_R2GRID_H__
