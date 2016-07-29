/* R3grid.cpp
 * R3 under discretization (discretize functor) to a grid
 * Ernest Yeung  ernestyalumni@gmail.com
 * 20160630
 */
#include "R2grid.h"


Grid2d :: Grid2d(std::array<int,2> Ld_in, std::array<float,2> ld_in)
	: Ld(Ld_in), ld(ld_in)
{
	hd = { ld[0]/Ld[0], ld[1]/Ld[1]  };
	
	temperature = new float[ this->NFLAT() ];

}

std::array<float,2> Grid2d :: gridpt_to_space(std::array<int,2> index) {
	std::array<float,2> Xi { index[0]*hd[0], index[1]*hd[1]  } ;
	return Xi;
}

int Grid2d :: NFLAT() {
	return Ld[0]*Ld[1] ;

}	

int Grid2d :: flatten(const int i_x, const int i_y ) {
	return i_x+i_y*Ld[0]  ;
}


Grid2d::~Grid2d() {
	delete[] temperature;
}

float gaussian2d(float A, float c, std::array<float,2> x_0, std::array<float,2> x) {
	return A * expf( (-1.f)*
		( (x[0]-x_0[0])*(x[0]-x_0[0]) + (x[1]-x_0[1])*(x[1]-x_0[1])   ) /  
			(2.f * c*c) );
}

