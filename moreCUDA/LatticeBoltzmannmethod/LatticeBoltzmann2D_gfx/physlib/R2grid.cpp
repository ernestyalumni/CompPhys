/* R2grid.cpp
 * R2 under discretization (discretize functor) to a (staggered) grid
 * Ernest Yeung  ernestyalumni@gmail.com
 * 20161113
 */
#include "R2grid.h"

// Constructor
/* ---------------------------------------------------------------- */
/* Sets the initial values for velocity u, p                        */
/* ---------------------------------------------------------------- */
Grid2d :: Grid2d(std::array<int,2> Ld_in, std::array<float,2> ld_in)
	: Ld(Ld_in), ld(ld_in)
{
	hd = { ld[0]/static_cast<float>(Ld[0]), ld[1]/static_cast<float>(Ld[1])  };
	staggered_Ld = { Ld[0]+2, Ld[1]+2 };

	std::vector<float2> temp_u( this->NFLAT() );
	u2 = temp_u;

	std::vector<float> temp_rh( this->NFLAT() );
	rh = temp_rh;

}

Grid2d :: Grid2d(std::array<int,2> Ld_in, std::array<float,2> ld_in, 
					float UI, float VI)
	: Ld(Ld_in), ld(ld_in)
{
	hd = { ld[0]/static_cast<float>(Ld[0]), ld[1]/static_cast<float>(Ld[1])  };
	staggered_Ld = { Ld[0]+2, Ld[1]+2 };
	

	float2 uI { UI, VI };
	std::vector<float2> temp_u( this->NFLAT() , uI);
	u2 = temp_u;

	std::vector<float> temp_rh( this->NFLAT() );
	rh = temp_rh;

	
}

Grid2d :: Grid2d(std::array<int,2> Ld_in, std::array<float,2> ld_in, 
					float UI, float VI, float DENSITY)
	: Ld(Ld_in), ld(ld_in)
{
	hd = { ld[0]/static_cast<float>(Ld[0]), ld[1]/static_cast<float>(Ld[1])  };
	staggered_Ld = { Ld[0]+2, Ld[1]+2 };

	

	float2 uI { UI, VI };
	std::vector<float2> temp_u( this->NFLAT() , uI);
	u2 = temp_u;

	std::vector<float> temp_rh( this->NFLAT() , DENSITY);
	rh = temp_rh;

	
}



std::array<float,2> Grid2d :: gridpt_to_space(std::array<int,2> index) {
	std::array<float,2> Xi { index[0]*hd[0], index[1]*hd[1]  } ;
	return Xi;
}

int Grid2d :: NFLAT() {
	return Ld[0]*Ld[1] ;

}
	
// int Grid2d :: staggered_SIZE() - returns the staggered grid size
/* this would correspond to Griebel's notation of 
 * (imax+1)*(jmax+1)
 */
int Grid2d :: staggered_SIZE() {
	return (staggered_Ld[0] )*(staggered_Ld[1]);
}

int Grid2d :: flatten(const int i_x, const int i_y ) {
	return i_x+i_y*Ld[0]  ;
}

int Grid2d :: staggered_flatten(const int i_x, const int i_y ) {
	return i_x+i_y*(staggered_Ld[0])  ;
}


Grid2d::~Grid2d() {
	std::exit(0);
}


