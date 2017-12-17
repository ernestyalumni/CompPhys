*abridged versions*

`dev_R2grid.h`  

```  
#ifndef __DEV_R2GRID_H__
#define __DEV_R2GRID_H__

class Dev_Grid2d
{
	public:
		dim3 Ld;  // Ld.x,Ld.y = L_x, L_y or i.e. imax,jmax 

		float* p_arr ; 
		float* F_arr ; 
		float* G_arr ; 
		float* u_arr ; 
		float* v_arr ; 
	
		// Constructor
		/* --------------------------------------------------------- */
		/* Sets the initial values for velocity u, p                 */
		/* --------------------------------------------------------- */
		__host__ Dev_Grid2d( dim3 );

		// destructor
		__host__ ~Dev_Grid2d();

		__host__ int flatten(const int i_x, const int i_y ) ;
};

#endif // __DEV_R2GRID_H__  
```  

`u_p.h`  
```  
#ifndef __U_P_H__
#define __U_P_H__

/*------------------------------------------------------------------- */
/* Computation of tentative velocity field (F,G) -------------------- */
/*------------------------------------------------------------------- */

__global__ void compute_F(const float deltat, 
	const float* u, const float* v, float* F,
	const int imax, const int jmax, const float deltax, const float deltay,
	const float gamma, const float Re) ; 

__global__ void compute_G(const float deltat, 
	const float* u, const float* v, float* G,
	const int imax, const int jmax, const float deltax, const float deltay,
	const float gamma, const float Re) ;

/*------------------------------------------------------------------- */
/* Computation of the right hand side of the pressure equation ------ */
/*------------------------------------------------------------------- */

__global__ void compute_RHS( const float* F, const float* G, 
	float* RHS, 
	const int imax, const int jmax, 
	const float deltat, const float deltax, const float deltay);

/*------------------------------------------------------------------- */
/* SOR iteration for the Poisson equation for the pressure
/*------------------------------------------------------------------- */

__global__ void poisson_redblack( float* p, const float* RHS, 
	const int imax, const int jmax,
	const float deltax, const float deltay, 
	const float omega) ;

/*------------------------------------------------------------------- */
/* computation of residual */
/*------------------------------------------------------------------- */

__global__ void compute_residual( const float* p, const float* RHS, 
	const int imax, const int jmax,
	const float deltax, const float deltay, 
	float* residualsq_Array) ;
	
/*------------------------------------------------------------------- */
/* computation of new velocity values */
/*------------------------------------------------------------------- */

__global__ void calculate_u( float* u, const float* p, const float* F, 
	const int imax, const int jmax, const float deltat, const float deltax ) ;


__global__ void calculate_v( float* v, const float* p, const float* G, 
	const int imax, const int jmax, const float deltat, const float deltay ) ;

#endif // __U_P_H__
```  

`boundary.h`  

```  
#ifndef __BOUNDARY_H__
#define __BOUNDARY_H__

#include "../physlib/dev_R2grid.h"       // Dev_Grid2d

/* --------------------------------------------------------------- */
/* Setting the boundary conditions at the boundary strip.      	   */
// on the device GPU memory
/* --------------------------------------------------------------- */
void set_BConditions( Dev_Grid2d & dev_grid2d ) ;

__host__ void set_lidcavity_BConditions( Dev_Grid2d & );

////////////////////////////////////////////////////////////////////////

__host__ void set_horiz_press_BCs( Dev_Grid2d & ) ;

////////////////////////////////////////////////////////////////////////

__host__ void set_vert_press_BCs( Dev_Grid2d & ) ;

#endif // __BOUNDARY_H__
```  

# OpenGL  

`tex_anim2d.h`  

```  
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
		exitfunc(); }

	void initGLUT(int *argc, char **argv) {...}
	void initPixelBuffer() {...}
	void exitfunc() {...}	
};	

```  

`main.cu`  

```  
GPUAnim2dTex bitmap( L_X, L_Y );
GPUAnim2dTex* testGPUAnim2dTex = &bitmap;

void make_render(dim3 Ld_in, int iters_per_render_in, GPUAnim2dTex* texmap ) {
	uchar4 *d_out = 0;
	cudaGraphicsMapResources(1, &texmap->cuda_pixbufferObj_resource, 0);
	cudaGraphicsResourceGetMappedPointer((void **)&d_out, NULL,
		texmap->cuda_pixbufferObj_resource);

	for (int i = 0; i < iters_per_render_in; ++i) {
	...
	} // for loop, iters per render, END	
} // END make render

``` 

```  
#include <functional>

... 

std::function<void()> render = std::bind( make_render, dev_L2, iters_per_render, testGPUAnim2dTex);

std::function<void()> draw_texture = std::bind( make_draw_texture, L_X, L_Y);

void display() {
	render() ;
	draw_texture();
	glutSwapBuffers();
}

```  

