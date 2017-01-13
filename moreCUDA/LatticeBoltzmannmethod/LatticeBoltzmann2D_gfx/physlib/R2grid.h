/* R2grid.h
 * R2 under discretization (discretize functor) to a (staggered) grid
 * Ernest Yeung  ernestyalumni@gmail.com
 * 20161113
 */
#ifndef __R2GRID_H__
#define __R2GRID_H__

#include <array> // std::array
#include <vector> // std::vector
#include <cstdlib> // std::exit


class Grid2d
{
	public : 
		std::array<int,2> Ld;		// Ld[0],Ld[1] = L_x, L_y or i.e. imax,jmax 
		std::array<float,2> ld;  // xlength, ylength or xLength, yLength (i.e. Domain size (non-dimensional))
		std::array<float,2> hd;  // delx, dely or dx, dy
		std::array<int,2> staggered_Ld; // Ld[0]+2,Ld[1]+2 = L_x+2,L_y+2, or i.e. imax+2,jmax+2 

		///////////////////////////////////////////////////////////////
		// Physical quantities over Euclidean space R^2, \mathbb{R}^2
		///////////////////////////////////////////////////////////////
		// 
		// rh
		// u2
		//
		// \rho \in C^{\infty}(\mathbb{R}^2) \xrightarrow{ \text{ discretize } }
		//  rh \in (\mathbb{R}^+)^{ Ld[0] * Ld[1] }
			
		///////////////////////////////////////////////////////////////
		
		// You need to compile using nvcc compiler and possibly -x cu flag, to treat C++ code as CUDA code
		// vector for velocity
		std::vector<float2> u2 ;	  // of size staggered_SIZE, or i.e. (L_x+2)*(L_y+2)==(imax+2)*(jmax+2)


		// since size isn't known statically at compile-time, but until runtime, using std::vector
		// vector for pressure
		std::vector<float> rh ;	 // of size staggered_SIZE, or i.e. (L_x+2)*(L_y+2)==(imax+2)*(jmax+2)
		
		// Constructor
		/* --------------------------------------------------------- */
		/* Sets the initial values for velocity u, p                 */
		/* --------------------------------------------------------- */
		Grid2d(std::array<int,2> Ld_in, std::array<float,2> ld_in);
		Grid2d(std::array<int,2> Ld_in, std::array<float,2> ld_in, 
				float UI, float VI);
		Grid2d(std::array<int,2> Ld_in, std::array<float,2> ld_in, 
				float UI, float VI, float DENSITY);

		
		std::array<float,2> gridpt_to_space(std::array<int,2> );
		
		int NFLAT();

		// int Grid2d :: staggered_SIZE() - returns the staggered grid size
		/* this would correspond to Griebel's notation of 
		 * (imax+1)*(jmax+1)
		 */
		int staggered_SIZE();
		
		int flatten(const int i_x, const int i_y ) ;

		int staggered_flatten(const int i_x, const int i_y ) ;

	
		~Grid2d();	
};


#endif // __R2GRID_H__
