/* surfObj2d.cu
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
#include "surfObj2d.h"

// constructor
__host__ SurfObj2d::SurfObj2d( cudaArray* cuArr ) : 
	surfObj(0)
{
	// Specify surface
	memset(&(this->resDesc),0, sizeof(this->resDesc));
	(this->resDesc).resType = cudaResourceTypeArray;
	// END specify surface object parameters

	// Create the surface object
	(this->resDesc).res.array.array = cuArr; 
	checkCudaErrors(
		cudaCreateSurfaceObject(&(this->surfObj), &(this->resDesc)) );
	
}

// constructor
// Note: you're going to have to set the devPtr this texture object is associated to at a later time.
__host__ SurfObj2d::SurfObj2d(  ) : 
	surfObj(0)
{
	// Specify texture
	memset(&(this->resDesc),0, sizeof(this->resDesc));
	(this->resDesc).resType = cudaResourceTypeArray;
	// END specify surface object parameters

	// Create surface object
//	(this->resDesc).res.array.array = cuArr;
/*	checkCudaErrors( 
		cudaCreateTextureObject(&(this->texObj), &(this->resDesc), &(this->texDesc), NULL) );
	*/
}

// destructor
__host__ SurfObj2d::~SurfObj2d() {
	checkCudaErrors(
		cudaDestroySurfaceObject(this->surfObj) ) ;
	
}
