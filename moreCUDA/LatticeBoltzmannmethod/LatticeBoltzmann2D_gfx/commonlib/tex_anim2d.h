/* tex_anim2d.h
 * 2-dim. GPU texture animation 
 * Ernest Yeung  ernestyalumni@gmail.com
 * 20160720
 */
#ifndef __TEX_ANIM2D_H__
#define __TEX_ANIM2D_H__

#include <stdio.h>

#define GL_GLEXT_PROTOTYPES // needed for identifier glGenBuffer, glBindBuffer, glBufferData, glDeleteBuffers

#include <GL/glut.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>


#include <stdlib.h> // exit
//#include <cstdlib> // exit

	// ################################################################
	// MANUALLY change minval, maxval in 
	// __global__ void float_to_char 
	// ################################################################


extern int iterationCount  ;

struct GPUAnim2dTex {
	GLuint pixbufferObj ; // OpenGL pixel buffer object
	GLuint texObj       ; // OpenGL texture object	

	cudaGraphicsResource *cuda_pixbufferObj_resource;
 
	int width, height;
 
	GPUAnim2dTex( int w, int h ) {
		width  = w;
		height = h;

		pixbufferObj = 0 ;
		texObj       = 0 ;
	}

	~GPUAnim2dTex() {
		exitfunc();
	}

	void initGLUT(int *argc, char **argv) {
		glutInit(argc, argv);
		glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
		glutInitWindowSize(width, height);
		glutCreateWindow("Velocity. Vis.");

		gluOrtho2D(0, width, height, 0);
	}
	
	void initPixelBuffer() {
		glGenBuffers(1, &pixbufferObj );
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pixbufferObj);  // glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pixbufferObj);
		
		glBufferData(GL_PIXEL_UNPACK_BUFFER, width*height*sizeof(GLubyte)*4, 0, 
			GL_STREAM_DRAW);
		
		glGenTextures(1, &texObj);
		glBindTexture(GL_TEXTURE_2D, texObj );
		
		// following 3 lines were not originally included
//		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
//		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
//		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
				
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		cudaGraphicsGLRegisterBuffer( &cuda_pixbufferObj_resource, pixbufferObj, 
			cudaGraphicsMapFlagsWriteDiscard);
	}
	
	void exitfunc() {
		if (pixbufferObj) {
			cudaGraphicsUnregisterResource( cuda_pixbufferObj_resource);
			glDeleteBuffers(1, &pixbufferObj);
			glDeleteTextures(1, &texObj);
		}
	}
};	
	
// interactions

void keyboard_func( unsigned char key, int x, int y) ; 
	
void mouse_func( int button, int state, int x, int y ) ;

void idle();

void printInstructions() ;
	
// make* functions make functions to pass into OpenGL (note OpenGL is inherently a C API)

void make_draw_texture(int w, int h); 

__global__ void float_to_char( uchar4* , const float* , const int L_x, const int L_y);

// float2_to_char, input float2 * velocity array, pick out .x component, transform to a char
__global__ void float2_to_char( uchar4* , const float2* , const int L_x, const int L_y);


// from physical scalar values to color intensities on an OpenGL bitmap
__global__ void floatux_to_char( uchar4* dev_out, cudaSurfaceObject_t uSurf, const int L_x, const int L_y) ; 

#endif // # __TEX_ANIM2D_H__ 
