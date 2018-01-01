/**
 * 	@file 	ising_2dim.cpp
 * 	@brief 	Program to solve the 2-dim. Ising model with 0 external field.    
 * 	@ref	https://github.com/CompPhysics/ComputationalPhysics/blob/master/doc/Programs/LecturePrograms/programs/StatPhys/cpp/ising_2dim.cpp
 * https://github.com/CompPhysics/ComputationalPhysics/blob/cde9f3b1ee798c36c66794bdd332b030a2c82c5c/doc/Programs/LecturePrograms/programs/cppLibrary/lib.cpp
 * 	@details Program to solve the 2-dim. Ising model with 0 external field. 
 * 				The coupling constant J = 1
 * 				Boltzmann's constant = 1, temperature has thus dimension energy
 * 				Metropolis sampling is used.  Periodic boundary conditions  
 * 
 * COMPILATION TIP : -Wall warning all, -g debugger, gdb can step through 
 * g++ -Wall -g ising_2dim.cpp -o ising_2dim
 * */
 
#include <iostream>  
#include <fstream>  
#include <cmath> // std::exp, std::fabs 
#include <new> // std::nothrow

/* @fn periodic
 * @brief inline function for periodic boundary conditions  
 * 
 * */
inline int periodic(int i, int limit, int add) {
	return (i+limit+add) % (limit); 
}

/** 
 * @fn ran1
 * @ref https://github.com/CompPhysics/ComputationalPhysics/blob/cde9f3b1ee798c36c66794bdd332b030a2c82c5c/doc/Programs/LecturePrograms/programs/cppLibrary/lib.cpp
 * @brief double ran1(long *idum)
 * @details is a "Minimal" random number generator of Park and Miller 
 * (see Numerical reciple page 280) with Bays-Durham shuffle and 
 * added safeguards.  Call with idum a negative integer to initialize; 
 * thereafter, do not alter idum between successive deviates in a 
 * sequence.  RNMX should approximate the largest floating point value 
 * that is less than 1. 
 * The function returns a uniform deviate between 0.0 and 1.0
 * (exclusive of end-point values).  
 */ 
double ran1(long *idum) 
{
	constexpr const int IA {16807};
	constexpr const int IM {2147483647}; 
	constexpr const float AM { 1.0/IM }; 
	constexpr const int IQ {127773}; 
	constexpr const int IR {2836}; 
	constexpr const int NTAB {32}; 
	constexpr const int NDIV { (1+(IM-1)/NTAB ) } ; 
	constexpr const double EPS {1.2e-7}; 
	constexpr const double RNMX (1.0-EPS); 
	
	int 		j;
	long 		k;
	static long iy = 0;
	static long iv[NTAB]; 
	double 		temp;
	
	if (*idum <=0 || !iy) {
		if (-(*idum) < 1) { 
			*idum = 1; }
		else { *idum = -(*idum); }
		for (j = NTAB + 7; j >=0 ; j--) {
			k = (*idum)/IQ;
			*idum = IA*(*idum - k*IQ) - IR*k; 
			if (*idum <0) { 
				*idum += IM; }
			if (j <NTAB) { 
				iv[j] = *idum; 
			}
		} // END for loop for j 
		iy = iv[0]; 
	}
	k  		= (*idum)/IQ;
	*idum 	= IA*(*idum -k*IQ)-IR*k;
	if (*idum <0) { *idum += IM; }
	j 	= iy/NDIV;
	iy 	= iv[j]; 
	iv[j] = *idum;
	if ((temp=AM*iy) > RNMX) { return RNMX ; } 
	else { return temp; } 
};

// Function declarations (in the beginning)  

// Function to initialize energy and magnetization
void initialize(int, double, int **, double&, double&); 

// The Metropolis algorithm  
void Metropolis(int, long&, int **, double&, double&, double *);


void **matrix(int, int, size_t);
void free_matrix(void **); 



int main(int argc, char* argv[] ) 
{
	long idum;
	
	int **spin_matrix, n_spins, mcs; // mcs is the number of trials to run Metropolis algorithm 
	double w[17], average[5], initial_temp, final_temp, E, M, temp_step; 
	
	mcs = 100; 
	n_spins = 256;
	initial_temp = 100.;
	
	// Read in initial values such as size of lattice, temp, and cycles  
	
	spin_matrix = (int **) matrix(n_spins, n_spins, sizeof(int));  
	
	idum = -1; // random starting point  
	
	
	double temperature = initial_temp; 
	
	// initialize energy and magnetization 
	E = M = 0.; 
	
	// setup array for possible energy changes
	for (int de =-8; de <=8; de++) { w[de+8] = 0 ; }
	for (int de =-8; de <= 8; de+=4) { w[de+8] = std::exp(-de/temperature); }
	
	// initialize array for expectation values 
	for (int i =0; i<5; i++) { average[i] = 0.; } 
	initialize(n_spins, temperature, spin_matrix, E, M); 
	
	// start MonteCarlo computation
	for (int cycles = 1; cycles <= mcs; cycles++) { 
		Metropolis(n_spins, idum, spin_matrix, E, M, w) ; 
		// update expectation values
		average[0] += E; 	average[1] += E*E;
		average[2] += M;	average[3] += M*M; average[4] += std::fabs(M);
	}
}

/**
 * @fn initialize
 * @brief function to initialize energy, spin matrix, and magnetization 
 * */
void initialize(int n_spins, double temperature, int **spin_matrix, 
				double& E, double& M) 
{
	// setup spin matrix and initial magnetization
	for (int y=0; y < n_spins; y++) {
		for (int x=0; x < n_spins; x++) {
			spin_matrix[y][x] = 1; // spin orientation for the ground state 
			M += (double) spin_matrix[y][x]; 
		}
	}
	// setup initial energy
	for (int y=0; y<n_spins; y++) {
		for (int x=0; x<n_spins; x++) {
			E -= (double) spin_matrix[y][x] * 
					(spin_matrix[periodic(y,n_spins,-1)][x] + 
					spin_matrix[y][periodic(x,n_spins,-1)]);
		}
	}
} // end of function initialize 


/**
 * @fn Metropolis
 * 
 * */
void Metropolis(int n_spins, long& idum, int **spin_matrix, double& E, double& M, double *w) 
{
	// loop over all spins, pick a random spin each time 
	for (int y = 0; y < n_spins; y++) {
		for (int x = 0; x <n_spins; x++) {
			// pick random spin 
			int ix = (int) (ran1(&idum)*(double)n_spins);
			int iy = (int) (ran1(&idum)*(double)n_spins); 

			// essentially a stencil operation
			int deltaE = 2*spin_matrix[iy][ix] *
				(spin_matrix[iy][periodic(ix,n_spins,-1)] + 
				spin_matrix[periodic(iy,n_spins,-1)][ix] + 
				spin_matrix[iy][periodic(ix,n_spins,1)] + 
				spin_matrix[periodic(iy,n_spins,1)][ix]); 
			if (ran1(&idum) <= w[deltaE+8]) {
				spin_matrix[iy][ix] *= -1; // flip 1 spin and accept new spin config 
				M += (double) 2*spin_matrix[iy][ix] ; 
				E += (double) deltaE; 
			}
		} // END of for loop x 
	} // END of for loop y 
} // end of Metropolis sampling over spins  



/**
 * @fn void **matrix
 * @brief The function 
 * 				void ** matrix()
 * 			reserves dynamic memory for a 2-dim. matrix 
 * 			using the C++ command new.  No initialization of the elements.  
 * Input data: 
 * 	int row 		- number of rows
 * 	int col 		- number of columns
 * 	int num_bytes 	- number of bytes for each  
 * 						element
 * @ref https://stackoverflow.com/questions/131803/unsigned-int-vs-size-t
 * @details The size_t type is the unsigned integer type 
 * that is the result of the sizeof operator (and the offsetof operator), 
 * so it is guaranteed to be big enough to contain 
 * the size of the biggest object your system can handle (e.g., a static array of 8Gb).
 * 
 * The size_t type may be bigger than, equal to, or smaller than an unsigned int, 
 * and your compiler might make assumptions about it for optimization.
 * 
 * @return Returns a void **pointer to the reserved memory location. 
 * */
//void **matrix(int row, int col, int num_bytes) {
void **matrix(int row, int col, size_t num_bytes) {
	int 	i, num; 
	int 	**pointer, *ptr; 
	
	pointer = new(std::nothrow) int*[row]; 
	if (!pointer) { 
		std::cout << "Exception handling: Memory allocation failed"; 
		std::cout << " for " << row << " row addresses ! " << std::endl; 
		return NULL;
	}
	i = (row * col * num_bytes)/sizeof(char) ;
	pointer[0] = new(std::nothrow) int [i]; 
	if (!pointer[0]) { 
		std::cout << "Exception handling: Memory allocation failed"; 
		std::cout << " for address to " << i << " characters ! " << std::endl; 
		return NULL; 
	}
	ptr = pointer[0]; 
	num = col * num_bytes; 
	for (i=0; i < row; i++, ptr += num) { 
		pointer[i] = ptr;
	}
	
	return (void**) pointer; 
} // end: function void **matrix() 

 
/**
 * @fn void free_matrix(void **matr)  
 * @details releases the memory reserved by the function matrix() for t 
 * derivatives dydx[1:n] and uses the fourth-order Runge-Kutta method to 
 * advance the solution over an interval h and return incremented variables 
 * as yout[1:n], which not need to be a distinct array from y[1:n].  The 
 * users suppy the routine derivs(x,y,dydx), which returns the derivatives
 * dydx at x. 
 * */ 
void free_matrix(void **matr) 
{
	delete [] (char *) matr[0];
	delete [] matr; 
} // End : function free_matrix() 

