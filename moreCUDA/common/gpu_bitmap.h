/*
 * gpu_bitmap_draft.h
 * make OpenGL bitmaps on the GPU
 * Ernest Yeung ernestyalumni@gmail.com
 * 20160616
 * */
#ifndef __GPU_BITMAP_H__
#define __GPU_BITMAP_H__

#define GL_GLEXT_PROTOTYPES /* needed for identifiers glBindBuffer, 
							 * glDeleteBuffer, glGenBuffers, glBufferData */
#include "GL/glut.h"
#include "cuda.h"
#include "cuda_gl_interop.h"

#include "errors.h"

struct GPUBitmap {
	GLuint bufferObj;
	cudaGraphicsResource *resource;
	int width, height;
	
	uchar4* devPtr;	
	size_t  size;
	
	GPUBitmap( int w, int h ) {
		width = w;
		height = h;
		
	/* select CUDA device to run application; need to know CUDA device ID so
	 * we can tell CUDA runtime that we intend to use device for CUDA AND OpenGL */
		cudaDeviceProp prop;
		int dev;
	
		memset( &prop, 0, sizeof( cudaDeviceProp) );
		prop.major = 2; // major version set to 2
		prop.minor = 1; // minor version set to 1
		HANDLE_ERROR( 
			cudaChooseDevice( &dev, &prop) ); /* instructs runtime to select GPU
											   * that satisfies constraints specified by
											   * cudaDeviceProp */
		HANDLE_ERROR( 
			cudaGLSetGLDevice( dev ) );
		/* We've prepared our CUDA runtime to play nicely with OpenGL driver with 
		 * cudaGLSetGLDevice */ 
		
		// initialize OpenGL driver by calling GL Utility Toolkit (GLUT)	
		// these GLUT calls need to be made before the other GL calls
		// soo and foo are "boiler plate"
		int soo = 1;
		char* foo = '\0';
		glutInit( &soo, &foo );
		glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGBA );
		glutInitWindowSize( width, height );
		glutCreateWindow( "bitmap" ); // to draw our results
		
		// create pixel buffer object in OpenGL, store in our global variable GLuint bufferObj
		glGenBuffers( 1, &bufferObj );  // generate buffer handle
		glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, bufferObj ); // bind buffer handle to pixel buffer
		glBufferData( GL_PIXEL_UNPACK_BUFFER_ARB, width * height * 4,
					  NULL, GL_DYNAMIC_DRAW_ARB ); /* request OpenGL driver to allocate buffer
													* enumerant GL_DYNAMIC_DRAW_ARB indicate buffer will be
													* modified repeatedly by application 
													* since we have no data to preload buffer with,
													* pass NULL */
		HANDLE_ERROR(
			cudaGraphicsGLRegisterBuffer( &resource, bufferObj, cudaGraphicsMapFlagsNone )); 

		HANDLE_ERROR(
			cudaGraphicsMapResources( 1, &resource, NULL ) );
		HANDLE_ERROR(
			cudaGraphicsResourceGetMappedPointer( (void**)&devPtr, 
												&size,
												resource ));
		
	}
	
	~GPUBitmap() {
		free_resources();
	}
	
	void free_resources( void ) {
		// clean up OpenGL and CUDA
		HANDLE_ERROR(
			cudaGraphicsUnregisterResource( resource )) ;
		glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, 0 );
		glDeleteBuffers( 1, &bufferObj );
	}

		
	static GPUBitmap** get_bitmap_ptr( void ) {
		static GPUBitmap *gBitmap;
		return &gBitmap;
	}
	
/* notice glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, bufferObj);
 * this call bounds shared buffer as pixel source for OpenGL driver to use, subsequently,
 * to glDrawPixels
 * */	
	static void draw_func(void) {
		GPUBitmap* bitmap = *(get_bitmap_ptr() );
		glDrawPixels( bitmap->width, bitmap->height, GL_RGBA, GL_UNSIGNED_BYTE, 0 );

		glutSwapBuffers();
	}
	
	static void key_func( unsigned char key, int x, int y) {
		switch (key) {
			case 27:
				GPUBitmap* bitmap = *(get_bitmap_ptr() );
				bitmap->free_resources();
				exit(0);
		}
	}

		
	
	void display_and_exit(void ) {
		GPUBitmap** bitmap = get_bitmap_ptr();
		*bitmap = this;
		/* cf. http://en.cppreference.com/w/cpp/language/this this pointer this is important */

		/* unmap our shared resource 
		 * this call is important, prior to performing rendering, because
		 * it provides synchornization between CUDA and graphics portions of application 
		 * it implies that all CUDA operations performed prior to call 
		 * cudaGraphicsUnmapResources() will complete before ensuing graphics calls begin */
		HANDLE_ERROR(
			cudaGraphicsUnmapResources( 1, &((*bitmap)->resource), NULL ));
				
		// set up GLUT and kick off main loop
		glutDisplayFunc( draw_func );
		glutKeyboardFunc( key_func);
		glutMainLoop();
		
	}
};
#endif // __GPU_BITMAP_H__
