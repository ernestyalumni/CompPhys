// diffusion1dexplicit.cpp
/* Explicit method for 1-dim. diffusion equation.
 * While based on 10.2.1 Explicit Scheme of Hjorth-Jensen (2015), see my 
 * reformulation because it's more general in CompPhys.pdf, 
 * same CompPhys/cpp github repository.
 * Uses gsl for linear algebra
 * */
/*
 * Compilation tip
 * 
 * g++ -lgsl -lgslcblas diffusion1dexplicit.cpp -o diffusion1dexplicit
 * 
 * */

#include <iostream>
#include <iomanip>  // std::setiosflags
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_math.h> // M_PI
#include <gsl/gsl_blas.h> // gsl_blas_dgemv

void createATrans(int, double, double, gsl_matrix* );

void explicitIteration(gsl_matrix *, gsl_vector *, gsl_vector *);

int main(int argc, char * argv[]) {
	// physical constants
	double l_x = 1.0; 
	
	// discretization parameters
	int L_x = 1000;
	double dx = l_x / ( static_cast<double>( L_x ) );
	double dt = 1.0e-7;
//	double tolerance = 1.0e-12;

	int Titers = 100;
	
	// "CFL" value-print out the so-called Courant-Friedrichs-Lewy parameter as sanity check
	double CFL_const = dt/(dx*dx);
	std::cout << "This is the CFL (Courant-Friedrichs-Lewy) value : " << CFL_const << 
		" for dx : " << dx << " and for dt : " << dt << std::endl ; 
	
	gsl_matrix* A = gsl_matrix_calloc(L_x-2,L_x-2); // initializes all elements of matrix to 0 
	
	// initial conditions
	gsl_vector* u0 = gsl_vector_calloc(L_x-2); // vector size L_x-2 is for the special case of the 
		// simple boundary condition of 0 at both ends
	for (int i=0; i < (L_x-2); ++i) {
		gsl_vector_set( u0, i, sin( M_PI/l_x*dx*i)) ; }
	
	// A - matrix A for the (time-evolution) transformation 	
	createATrans( L_x-2, dx, dt, A); 	
	
	// Start the iterative solver 
	gsl_vector* u_new = gsl_vector_calloc(L_x-2);

	for (int j = 0; j < Titers; j++) {
		explicitIteration( A, u0, u_new) ;
	} 
	
	// Testing against exact solution
	double ExactSolution = 0.0;
	double sum = 0.0;
	for (int i =0; i < (L_x-2); ++i) {
		ExactSolution = sin( M_PI/l_x*dx*i)*exp( (-1.0)*( M_PI/l_x)*(M_PI/l_x)*Titers*dt);
		sum += fabs((gsl_vector_get(u0,i) - ExactSolution)); 
	}
	
	std::cout << std::setprecision(5) << std::setiosflags(std::ios::scientific);
	std::cout << "Explicit (matrix) method: L2Error is " << sum/L_x << 
					" in " << Titers << " iterations " << std::endl;
	
	
	
	gsl_matrix_free(A);
	gsl_vector_free(u0);
	gsl_vector_free(u_new);
}

void createATrans(int MatrixSIZE, double dx, double dt, gsl_matrix *A ) {
	double alpha_const = dt/(dx*dx);
	
	for (int i = 1; i < MatrixSIZE-1; i++) {
		gsl_matrix_set( A, i, i-1, alpha_const ) ; 
		gsl_matrix_set( A, i, i+1, alpha_const ) ; 
		gsl_matrix_set( A, i, i, 1.0 - 2.0*alpha_const ); } 
	
	// set how transformation A deals with "the simple" boundary condition (0 at both ends)
	gsl_matrix_set( A, 0, 0, 1.0-2.0*alpha_const) ; 
	gsl_matrix_set( A, 0, 1, alpha_const) ; 
	gsl_matrix_set( A, MatrixSIZE-1, MatrixSIZE-2, 1.0-2.0*alpha_const) ; 
	gsl_matrix_set( A, MatrixSIZE-1, MatrixSIZE-1, alpha_const) ; 
		
		
}



void explicitIteration(gsl_matrix *A, gsl_vector *u, gsl_vector *u_new ) {
	gsl_blas_dgemv( CblasNoTrans, 1.0, A, u, 0.0, u_new) ; 
	gsl_vector_swap( u, u_new );
	
}
