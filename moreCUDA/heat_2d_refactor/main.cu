/* main.cpp
 * 2-dim. Laplace Eq. (heat eq.) by finite difference with shared memory
 * Ernest Yeung  ernestyalumni@gmail.com
 * 20160625
 */
#include <functional>

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "./physlib/heat_2d.h"
#include "./physlib/dev_R2grid.h"
#include "./physlib/R2grid.h"

#include "./commonlib/errors.h"
#include "./commonlib/tex_anim2d.h"
#include "./commonlib/finitediff.h"


#define GL_GLEXT_PROTOTYPES // needed for identifier glGenBuffer, glBindBuffer, glBufferData, glDeleteBuffers

#include <GL/glut.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h> // or #include "cuda_gl_interop.h"
#define ITERS_PER_RENDER 50

const float Deltat[1] { 0.0001f };

// physics
const int WIDTH  { 640 } ;
const int HEIGHT { 640 } ;

dim3 dev_L2 { static_cast<unsigned int>(WIDTH), 
				static_cast<unsigned int>(HEIGHT) };

dev_Grid2d dev_grid2d( dev_L2 );		

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
//		kernelLauncher2(d_out, dev_grid2d.dev_temperature, w, h, bc, M_i );
//		kernelLauncher(d_out, dev_grid2d.dev_temperature, w, h, bc, M_i );
//		kernelLauncher3(d_out, dev_grid2d.dev_temperature, w, h, bc, M_i );
		kernelLauncher4(d_out, dev_grid2d.dev_temperature, w, h, bc, M_i );
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
	// physics
	constexpr std::array<int,2> LdS {WIDTH, HEIGHT };
	constexpr std::array<float,2> ldS {1.f, 1.f };
	
	HANDLE_ERROR(
		cudaMemcpyToSymbol( dev_Deltat, Deltat, sizeof(float)*1,0,cudaMemcpyHostToDevice) );

	const float heat_params[2] { 
//								 0.0061035f , 
								 0.00061035f,
								 1.f } ; // \kappa 
										// heat capacity for constant volume, per volume 

	HANDLE_ERROR(
		cudaMemcpyToSymbol( dev_heat_params, heat_params, sizeof(float)*2,0,cudaMemcpyHostToDevice) );
	
	const int Ld_to_const[3] { LdS[0], LdS[1] } ;
	
	HANDLE_ERROR(
		cudaMemcpyToSymbol( dev_Ld, Ld_to_const, sizeof(int)*2,0,cudaMemcpyHostToDevice) );
	
	Grid2d grid2d( LdS, ldS);
	
	const float hds[2] { grid2d.hd[0], grid2d.hd[1] } ;
	
	// sanity check
	std::cout << " hds : .x : " << hds[0] << " .y : " << hds[1] << std::endl;
	
//	set1DerivativeParameters(hds);
//	set2DerivativeParameters(hds);
//	set3DerivativeParameters(hds);
	set4DerivativeParameters(hds);
	
	
	resetTemperature( dev_grid2d.dev_temperature, WIDTH, HEIGHT, bc, M_i);
	
	
	printInstructions();


	testGPUAnim2dTex->initGLUT(&argc, argv);

	glutKeyboardFunc(keyboard_func);
	glutMouseFunc(mouse_func);
	glutIdleFunc(idle);

	glutDisplayFunc(display);

	testGPUAnim2dTex->initPixelBuffer();

	glutMainLoop();

	HANDLE_ERROR(
		cudaFree( dev_grid2d.dev_temperature ) );

	return 0;
} 

	
