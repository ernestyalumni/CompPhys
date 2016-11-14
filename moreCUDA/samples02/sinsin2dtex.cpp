/* sinsin2dtex.cpp
 * sine * sine function over 2-dimensional textured grid
 * Ernest Yeung  ernestyalumni@gmail.com
 * 20161113
 * 
 * Compilation tips if you're not using a make file
 * 
 * g++ -std=c++11 -c ./physlib/R2grid.cpp -o R2grid.o
 * g++ -std=c++11 sinsin2dtex.cpp R2grid.o -o sinsins2tex
 * 
 */
#include <iostream> // std::cout

#include "./physlib/R2grid.h"

constexpr const int WIDTH  { 640 } ;
constexpr const int HEIGHT { 640 } ;

int main() {
	constexpr const int DISPLAY_SIZE { 22 };

	constexpr const float PI { acos(-1.f) };
	
	constexpr std::array<int,2> LdS {WIDTH, HEIGHT };
	constexpr std::array<float,2> ldS {1.f, 1.f };

	Grid2d grid2d( LdS, ldS);
	
	// sanity check
	std::cout << "This is the value of pi : " << PI << std::endl;
	std::cout << "Size of rho on grid2d?  Do grid2d.rho.size() : " << grid2d.rho.size() << std::endl;
	std::cout << "Initially, on grid2d.rho : " << std::endl;
	for (auto i = 0; i < DISPLAY_SIZE; ++i ) {
		std::cout << " " << grid2d.rho[i] ; }
	std::cout << std::endl;

	std::array<int,2> ix_in { 0, 0 };
	std::array<float,2> Xi { 0.f, 0.f };
	for (auto j = 0; j < grid2d.Ld[0] ; ++j ) { 
		for (auto i = 0; i < grid2d.Ld[1] ; ++i ) {
			ix_in[0] = i ;
			ix_in[1] = j ;
			Xi = grid2d.gridpt_to_space( ix_in );	
			grid2d.rho[ grid2d.flatten(i,j) ] = sin( 2.f*PI*Xi[0])*sin(2.f*PI*Xi[1]) ; 
		}
	}

	// sanity check
	std::cout << "grid2d.rho, after initializing with values given by sin(2*pi*x)*sin(2*pi*y) : " << std::endl;
	for (auto i = 0; i < DISPLAY_SIZE; ++i ) {
		std::cout << " " << grid2d.rho[(i+WIDTH/4)+ HEIGHT/4*WIDTH] ; }
	std::cout << std::endl;

	
	return 0;
}
