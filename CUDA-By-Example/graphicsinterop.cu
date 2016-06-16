/* cf. Jason Sanders, Edward Kandrot. CUDA by Example: An Introduction to General-Purpose GPU Programming */
/* 
** Chapter 8 Graphics Interoperability 
** 8.2 Graphics Interoperation
*/
#define DIM 2048

#define GL_GLEXT_PROTOTYPES
#include "GL/glut.h"
#include "cuda.h"
#include "cuda_gl_interop.h"
#include "./common/errors.h"

#include <cmath> 
  

/* global variables bufferObj, resource
 * stores different handles to the same buffer; OpenGL and CUDA will both have different "names" for
 * same buffer
 * */
GLuint bufferObj;
cudaGraphicsResource *resource;

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

/* notice glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, bufferObj);
 * this call bounds shared buffer as pixel source for OpenGL driver to use, subsequently,
 * to glDrawPixels
 * */
static void draw_func(void) {
	glDrawPixels( DIM, DIM, GL_RGBA, GL_UNSIGNED_BYTE, 0 );
	glutSwapBuffers();
}

static void key_func( unsigned char key, int x, int ) {
	switch (key) {
		case 27:
		// clean up OpenGL and CUDA
		HANDLE_ERROR(
			cudaGraphicsUnregisterResource( resource )) ;
			glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, 0 );
			glDeleteBuffers( 1, &bufferObj );
			exit(0);
	}
}

int main( int argc, char **argv) {
	/* select CUDA device to run application; need to know CUDA device ID so
	 * we can tell CUDA runtime that we intend to use device for CUDA AND OpenGL */
	cudaDeviceProp prop;
	int dev;
	
	memset( &prop, 0, sizeof( cudaDeviceProp) );
	prop.major = 1; // major version set to 1
	prop.minor = 0; // minor version set to 0
	HANDLE_ERROR( 
		cudaChooseDevice( &dev, &prop) ); /*  instructs runtime to select GPU
											* that satisfies constraints specified by
											* cudaDeviceProp */
	HANDLE_ERROR( 
		cudaGLSetGLDevice( dev ) );
	/* We've prepared our CUDA runtime to play nicely with OpenGL driver with 
	 * cudaGLSetGLDevice */ 
	
	// initialize OpenGL driver by calling GL Utility Toolkit (GLUT)	
	// these GLUT calls need to be made before the other GL calls
	glutInit( &argc, argv );
	glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGBA );
	glutInitWindowSize( DIM, DIM );
	glutCreateWindow( "bitmap" ); // to draw our results
	
	// create pixel buffer object in OpenGL, store in our global variable GLUint bufferObj
	glGenBuffers( 1, &bufferObj );  // generate buffer handle
	glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, bufferObj ); // bind buffer handle to pixel buffer
	glBufferData( GL_PIXEL_UNPACK_BUFFER_ARB, DIM * DIM * 4,
				  NULL, GL_DYNAMIC_DRAW_ARB ); /* request OpenGL driver to allocate buffer
												* enumerant GL_DYNAMIC_DRAW_ARB indicate buffer will be
												* modified repeatedly by application 
												* since we have no data to preload buffer with,
												* pass NULL */
	// notify CUDA runtime we intend to share OpenGL buffer bufferObj with CUDA
	HANDLE_ERROR(
		cudaGraphicsGLRegisterBuffer( &resource, bufferObj, cudaGraphicsMapFlagsNone )); /*
		* CUDA runtime returns CUDA-friendly handle to buffer, OpenGL pixel buffer object (PBO),
		* bufferObj in variable resource 
		* flag cudaGraphicsMapFlagsNone specifies that there's no particular behavior of this buffer
		* we'd want to specify
		* options: cudaGraphicsMapFlagsReadOnly - buffer will be read only
		* cudaGraphicsMapFlagsWriteDiscard - previous contents will be discarded, making
		* buffer essentially writeonly
		* flags allow CUDA and OpenGL drivers to optimize hardware settings for buffers, although
		* not required to be set
		* */
	
	/* instruct CUDA runtime to map shared resource and then by requesting pointer to mapped resource
	 * */
	uchar4* devPtr;
	size_t size; // unsigned integer type, to represent size of an object 
	HANDLE_ERROR(
		cudaGraphicsMapResources( 1, &resource, NULL ) );
	HANDLE_ERROR(
		cudaGraphicsResourceGetMappedPointer( (void**)&devPtr, 
												&size,
												resource ));
	/* we can then use devPtr as we would any device pointer, 
	 * except data can also be used by OpenGL as a pixel source */								
	
	dim3 grids(DIM/16,DIM/16);
	dim3 threads(16,16);
	kernel<<<grids,threads>>>( devPtr );
	
	HANDLE_ERROR(
		cudaGraphicsUnmapResources( 1, &resource, NULL ));
		
	// set up GLUT and kick off main loop
	glutKeyboardFunc( key_func );
	glutDisplayFunc( draw_func );
	glutMainLoop();
		
};
												
	
	
