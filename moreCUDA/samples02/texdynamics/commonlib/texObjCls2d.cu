/* texObjCls2d.cu
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
#include "texObjCls2d.h"

// constructor
__host__ TexObj2d::TexObj2d( float* buffer, const int L ) : 
	texObj(0)
{
	// Specify texture
	memset(&(this->resDesc),0, sizeof(this->resDesc));
	(this->resDesc).resType = cudaResourceTypeLinear;
	(this->resDesc).res.linear.devPtr = buffer ;
	(this->resDesc).res.linear.desc.f = cudaChannelFormatKindFloat;  
	(this->resDesc).res.linear.desc.x = 32; // bits per channel
	(this->resDesc).res.linear.sizeInBytes = sizeof(float) * L;
	

	// Specify texture object parameters
	memset(&(this->texDesc), 0, sizeof(this->texDesc) );

/* in M. Clark's implementation, I didn't see these "modes", addressMode, filterMode, being explicitly set. 
 * Indeed, for the Texture Reference API implementations prior, they weren't set.  
 * Nevertheless, I'll try to set them because for cuArray, they were set.
 */
	// set these 2 lines manually, either cudaAddressModeClamp or cudaAddressModeWrap
/*	(this->texDesc).addressMode[0] = cudaAddressModeClamp ;
	(this->texDesc).addressMode[1] = cudaAddressModeClamp ;
	// set filterMode manually; either cudaFilterModePoint or cudaFilterModeLinear
	(this->texDesc).filterMode     = cudaFilterModePoint ;
*/
// END of "optional" mode settings

	
	(this->texDesc).readMode       = cudaReadModeElementType;
	// END specify texture object parameters

	// Create texture object
	checkCudaErrors(
		cudaCreateTextureObject(&(this->texObj), &(this->resDesc), &(this->texDesc), NULL) );
	
}

// constructor
// Note: you're going to have to set the devPtr this texture object is associated to at a later time.
__host__ TexObj2d::TexObj2d(  ) : 
	texObj(0)
{
	// Specify texture
	memset(&(this->resDesc),0, sizeof(this->resDesc));
	(this->resDesc).resType = cudaResourceTypeLinear;
	(this->resDesc).res.linear.desc.f = cudaChannelFormatKindFloat;  
	(this->resDesc).res.linear.desc.x = 32; // bits per channel

	// Specify texture object parameters
	memset(&(this->texDesc), 0, sizeof(this->texDesc) );

/*
	// set these 2 lines manually, either cudaAddressModeClamp or cudaAddressModeWrap
	(this->texDesc).addressMode[0] = cudaAddressModeClamp ;
	(this->texDesc).addressMode[1] = cudaAddressModeClamp ;
	// set filterMode manually; either cudaFilterModePoint or cudaFilterModeLinear
	(this->texDesc).filterMode     = cudaFilterModePoint ;
*/
	
	(this->texDesc).readMode       = cudaReadModeElementType;
	// END specify texture object parameters

	// Create texture object
	checkCudaErrors( 
		cudaCreateTextureObject(&(this->texObj), &(this->resDesc), &(this->texDesc), NULL) );
	
}

// destructor
__host__ TexObj2d::~TexObj2d() {
	checkCudaErrors(
		cudaDestroyTextureObject(this->texObj) ) ;
	
}
