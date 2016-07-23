/* heat_2d.h
 * 2-dim. Laplace eq. (heat eq.) by finite difference with shared memory
 * Ernest Yeung  ernestyalumni@gmail.com
 * 20160626
 */
#ifndef __HEAT_2D_H__
#define __HEAT_2D_H__

struct uchar4;
// struct BC that contains all the boundary conditions
typedef struct {
	int x, y; // x and y location of pipe center
	float rad; // radius of pipe
	int chamfer; // chamfer
	float t_s, t_a, t_g; // temperatures in pipe, air, ground
} BC;


void kernelLauncher(uchar4 *d_out, float *d_temp, int w, int h, BC bc, dim3 M_in) ; 

void resetTemperature(float *d_temp, int w, int h, BC bc, dim3 M_in);

#endif // __HEAT_2D_H__
 
