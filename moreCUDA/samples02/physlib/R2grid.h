/* R2grid.h
 * R2 under discretization (discretize functor) to a grid
 * Ernest Yeung  ernestyalumni@gmail.com
 * 20161113
 */
#ifndef __R2GRID_H__
#define __R2GRID_H__

#include <array> // std::array
#include <vector> // std::vector
#include <cstdlib> // std::exit
#include <cmath> // expf, acos

class Grid2d
{
	public : 
		std::array<int,2> Ld;
		std::array<float,2> ld;
		std::array<float,2> hd;

		// since size isn't known statically at compile-time, but until runtime, using std::vector
		std::vector<float> rho ;	
		std::vector<float> rho_out ;	

		
		Grid2d(std::array<int,2> Ld_in, std::array<float,2> ld_in);
		
		std::array<float,2> gridpt_to_space(std::array<int,2> );
		
		int NFLAT();
		
		int flatten(const int i_x, const int i_y ) ;
	
		~Grid2d();	
};

float gaussian2d(float A, float c, std::array<float,2> x_0, std::array<float,2> x);

#endif // __R2GRID_H__
