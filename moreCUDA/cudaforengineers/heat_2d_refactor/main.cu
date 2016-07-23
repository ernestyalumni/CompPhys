/* main.cpp
 * 2-dim. Laplace Eq. (heat eq.) by finite difference with shared memory
 * Ernest Yeung  ernestyalumni@gmail.com
 * 20160625
 */
#include <functional>

#include "heat_2d.h"
#include <stdio.h>
#include <stdlib.h>

#include "./commonlib/tex_anim2d.h"

#define GL_GLEXT_PROTOTYPES // needed for identifier glGenBuffer, glBindBuffer, glBufferData, glDeleteBuffers

#include <GL/glut.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h> // or #include "cuda_gl_interop.h"
#define ITERS_PER_RENDER 50

// physics
const int WIDTH  { 640 } ;
const int HEIGHT { 640 } ;

float *d_temp = 0;

// graphics + physics

const dim3 M_i { 32 , 32  };

const int iters_per_render { 50 };

GPUAnim2dTex bitmap( WIDTH, HEIGHT );
GPUAnim2dTex* testGPUAnim2dTex = &bitmap; 


void make_render( int w, int h, int iters_per_render_in, GPUAnim2dTex* texmap  ) {
	uchar4 *d_out = 0;
	cudaGraphicsMapResources(1, &texmap->cuda_pixbufferObj_resource, 0);
	cudaGraphicsResourceGetMappedPointer((void **)&d_out, NULL,
		texmap->cuda_pixbufferObj_resource);

	for (int i = 0; i < iters_per_render_in; ++i) {
		kernelLauncher(d_out, d_temp, w, h, bc, M_i );
	}

	cudaGraphicsUnmapResources(1, &texmap->cuda_pixbufferObj_resource, 0);
	
	char title[128];
	sprintf(title, "Temperature Visualizer - Iterations=%4d, "
				   "T_s=%3.0f, T_a=%3.0f, T_g=%3.0f",
				   iterationCount, bc.t_s, bc.t_a, bc.t_g);
	glutSetWindowTitle(title);
}	



std::function<void()> render = std::bind( make_render, WIDTH, HEIGHT, iters_per_render, testGPUAnim2dTex);	

std::function<void()> draw_texture = std::bind( make_draw_texture, WIDTH, HEIGHT) ;

void display() {
	render();
	draw_texture();
	glutSwapBuffers();
}



int main(int argc, char** argv) {

	cudaMalloc(&d_temp, WIDTH*HEIGHT*sizeof(float));
	resetTemperature(d_temp, WIDTH, HEIGHT, bc, M_i);
	printInstructions();


	testGPUAnim2dTex->initGLUT(&argc, argv);

	glutKeyboardFunc(keyboard_func);
	glutMouseFunc(mouse_func);
	glutIdleFunc(idle);

	glutDisplayFunc(display);

	testGPUAnim2dTex->initPixelBuffer();

	glutMainLoop();

	cudaFree(d_temp);

	return 0;
} 

	
