/* cf. Jason Sanders, Edward Kandrot. CUDA by Example: An Introduction to General-Purpose GPU Programming  
 * Chapter 8 Graphics Interoperability 
 * 8.3 GPU Ripple with Graphics Interoperability
 */
#ifndef __GPU_ANIM_H__
#define __GPU_ANIM_H__

#define GL_GLEXT_PROTOTYPES // needed for identifier glGenBuffer, glBindBuffer, glBufferData, glDeleteBuffers

#include <GL/glut.h>
// #include <GL/glext.h>
// #include <GL/glx.h>
#include "cuda.h"
#include "cuda_gl_interop.h"
#include "errors.h"
#include <iostream>


struct GPUAnimBitmap {
	GLuint bufferObj;
	cudaGraphicsResource *resource;
	int  width, height;
	
	void *dataBlock;
	
	void (*fAnim)(uchar4*,void*,int);
	void (*animExit)(void*);
	void (*clickDrag)(void*,int,int,int,int);
	int  dragStartX, dragStartY;
	
	GPUAnimBitmap( int w, int h, void *d = NULL) {
		width     = w;
		height    = h;
		dataBlock = d;
		clickDrag = NULL;
	
		// first, find a CUDA device and set it to graphic interoperability
		cudaDeviceProp prop;
		int dev;
		memset( &prop, 0, sizeof( cudaDeviceProp ));
		prop.major = 2;
		prop.minor = 1;
		HANDLE_ERROR(
			cudaChooseDevice( &dev, &prop ));
		cudaGLSetGLDevice( dev );
	
		// trick GLUT into thinking we're passing an argument; hence soo and foo
		int soo = 1;
		char *foo = '\0';
		glutInit( &soo, &foo );
		glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGBA );
		glutInitWindowSize( width, height );
		glutCreateWindow( "bitmap" ); // to draw our results; "bitmap" name is arbitrary
	
		// create pixel buffer object in OpenGL, store in our global variable GLuint bufferObj
		glGenBuffers(1, &bufferObj);
		glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, bufferObj );
		glBufferData( GL_PIXEL_UNPACK_BUFFER_ARB, width * height * 4, 
					  NULL, GL_DYNAMIC_DRAW_ARB );
		HANDLE_ERROR(
			cudaGraphicsGLRegisterBuffer( &resource, bufferObj, cudaGraphicsMapFlagsNone ) );
	}
	
	~GPUAnimBitmap() {
		free_resources();
	}
	
	void free_resources( void ) {
	// clean up OpenGL and CUDA
		HANDLE_ERROR(
			cudaGraphicsUnregisterResource( resource )) ;
		glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, 0 );
		glDeleteBuffers( 1, &bufferObj );
	//			exit(0) ;
	}
	
	long image_size( void ) const { return width * height * 4 ; }
	
	static GPUAnimBitmap** get_bitmap_ptr( void ) {
		static GPUAnimBitmap* gBitmap;
		return &gBitmap;
	}
	
	// static method used for GLUT callbacks
	// maps shared buffer and retrieves a GPU pointer for this buffer
	static void idle_func( void ) {
		static int ticks = 1;
		GPUAnimBitmap* bitmap = *(get_bitmap_ptr() );
		uchar4*        devPtr;
		size_t size;
		
		HANDLE_ERROR(
			cudaGraphicsMapResources( 1, &(bitmap->resource), NULL) );
		HANDLE_ERROR(
			cudaGraphicsResourceGetMappedPointer( (void**)&devPtr, &size, bitmap->resource) );
			
		// fAnim() will launch CUDA C kernel to fill buffer at devPtr with image data.
		bitmap->fAnim( devPtr, bitmap->dataBlock, ticks++);
		
		// unmap GPU pointer that'll release buffer for use by OpenGL driver in rendering
		// this rendering is triggered by glutPostRedisplay() call
		HANDLE_ERROR(
			cudaGraphicsUnmapResources( 1, &(bitmap->resource), NULL ));
		glutPostRedisplay();
	};

	void anim_and_exit( void (*f)(uchar4*,void*,int), void(*e)(void*) ) {
		GPUAnimBitmap** bitmap = get_bitmap_ptr();
		*bitmap = this;
		fAnim = f;
		animExit = e;
		
		glutKeyboardFunc( key_func );
		glutDisplayFunc( draw_func );
		if (clickDrag != NULL )
			glutMouseFunc( mouse_func );
		glutIdleFunc( idle_func);
		glutMainLoop();
	}
	
	
	static void draw_func(void) {
		GPUAnimBitmap* bitmap = *(get_bitmap_ptr());
		glClearColor( 0.0, 0.0, 0.0, 1.0 );
		glClear( GL_COLOR_BUFFER_BIT );
		glDrawPixels( bitmap->width, bitmap->height, 
					  GL_RGBA, GL_UNSIGNED_BYTE, 0 );
		glutSwapBuffers();
	}
	
	static void key_func( unsigned char key, int x, int y) {
		switch (key) {
			case 27:
				GPUAnimBitmap* bitmap = *(get_bitmap_ptr() );
				if (bitmap->animExit)
					bitmap->animExit( bitmap->dataBlock );
				bitmap->free_resources();
				exit(0);
		}
	}
	
	void click_drag( void (*f)(void*,int,int,int,int)) {
		clickDrag = f;
	}
	
	static void mouse_func( int button, int state,
							int mx, int my) {
		if (button == GLUT_LEFT_BUTTON) {
			GPUAnimBitmap* bitmap = *(get_bitmap_ptr() );
			if (state == GLUT_DOWN) {
				bitmap->dragStartX = mx;
				bitmap->dragStartY = my;
			} else if (state == GLUT_UP) {
				bitmap->clickDrag(  bitmap->dataBlock,
									bitmap->dragStartX,
									bitmap->dragStartY,
									mx, my );
			}
		}
	}
};

#endif // __GPU_ANIM_H__
