/* texObjCls2d.cu
 * texture memory Object class, 2-dim. 
 * Ernest Yeung  ernestyalumni@gmail.com
 * 20161115
 *  
 * compilation tip: (compile separately)
 * nvcc -std=c++11 -c ./commonlib/texObjCls2d.cu -o texObjCls2d.o
 * 
 */
#include "texObjCls2d.h"

// constructor
__host__ TexObj2d::TexObj2d( cudaArray* cuArray ) : 
	texObj(0)
{
	// Specify texture
	memset(&(this->resDesc),0, sizeof(this->resDesc));
	(this->resDesc).resType = cudaResourceTypeArray;
	(this->resDesc).res.array.array = cuArray;

	// Specify texture object parameters
	memset(&(this->texDesc), 0, sizeof(this->texDesc) );

	// set these 2 lines manually, either cudaAddressModeClamp or cudaAddressModeWrap
	(this->texDesc).addressMode[0] = cudaAddressModeClamp ;
	(this->texDesc).addressMode[1] = cudaAddressModeClamp ;
	// set filterMode manually; either cudaFilterModePoint or cudaFilterModeLinear
	(this->texDesc).filterMode     = cudaFilterModePoint ;
	
	(this->texDesc).readMode       = cudaReadModeElementType;
	// END specify texture object parameters

	// Create texture object
	checkCudaErrors(
		cudaCreateTextureObject(&(this->texObj), &(this->resDesc), &(this->texDesc), NULL) );
	
}

// constructor
// Note: you're going to have to set the cudaArray this texture object is associated to at a later time.
__host__ TexObj2d::TexObj2d(  ) : 
	texObj(0)
{
	// Specify texture
	memset(&(this->resDesc),0, sizeof(this->resDesc));
	(this->resDesc).resType = cudaResourceTypeArray;

	// Specify texture object parameters
	memset(&(this->texDesc), 0, sizeof(this->texDesc) );

	// set these 2 lines manually, either cudaAddressModeClamp or cudaAddressModeWrap
	(this->texDesc).addressMode[0] = cudaAddressModeClamp ;
	(this->texDesc).addressMode[1] = cudaAddressModeClamp ;
	// set filterMode manually; either cudaFilterModePoint or cudaFilterModeLinear
	(this->texDesc).filterMode     = cudaFilterModePoint ;
	
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
