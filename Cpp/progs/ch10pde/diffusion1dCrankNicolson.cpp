// diffusion1dCrankNicolson.cpp
/* Crank-Nicolson method for 1-dim. diffusion equation, i.e. backward Euler.
 * While based on 10.2.3 Crank Nicolson of Hjorth-Jensen (2015), see my 
 * reformulation because it's more general in CompPhys.pdf, 
 * same CompPhys/cpp github repository.
 * Uses gsl for linear algebra
 * */
/*
 * Compilation tip
 * 
 * g++ -lgsl -lgslcblas diffusion1dimplicit.cpp -o diffusion1dimplicit
 * 
 * */
#include <iostream>
#include <iomanip> // std::setiosflags
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_math.h> // M_PI
#include <gsl/gsl_blas.h> // gsl_blas_dgemv
#include <gsl/gsl_linalg.h> // gsl_linalg_solve_symm_tridiag

void createATrans(int, double, double, gsl_matrix* );

void CrankNicolsonIteration(gsl_matrix *, 
							gsl_vector *, gsl_vector *, gsl_vector *, gsl_vector *, gsl_vector *) ;

int main(int argc, char * argv[]) {
	// physical constants
	double l_x = 1.0;
	
	// discretization parameters
	int L_x = 1000;
	double dx = l_x / (static_cast<double>(L_x));
	double dt = 1.0e-7;
	
	int Titers = 100;
	
	// "CFL" value-print out the so-called Courant-Friedrichs-Lewy parameter as sanity check
	double CFL_const = dt/(dx*dx);
	std::cout << "This is the CFL (Courant-Friedrichs-Lewy value : " << CFL_const << 
		" for dx : " << dx << " and for dt : " << dt << std::endl;
		
	// initial conditions
	gsl_vector* u0 = gsl_vector_calloc(L_x-2); /* vector size L_x-2 is for the special case of the 
												* simple boundary condition of 0 at both ends */
	for (int i=0; i<(L_x-2); ++i) {
		gsl_vector_set(u0, i, sin( M_PI/l_x*dx*i)) ; }
	
	
	// setting up the iterative solver
	gsl_vector* DiagonalEntries    = gsl_vector_calloc(L_x-2);
	gsl_vector* OffDiagonalEntries = gsl_vector_calloc(L_x-3);

	for (int i=0; i<(L_x-2); ++i) {
		gsl_vector_set(DiagonalEntries,i, 1.0 + CFL_const );
	}
	for (int i=0; i<(L_x-3); ++i) {
		gsl_vector_set(OffDiagonalEntries,i,(-1.0)*CFL_const/2.0 );
	}
	
	// A - matrix A for the "first part" of Crank-Nicolson method on previous time step values	
	gsl_matrix* A = gsl_matrix_calloc(L_x-2,L_x-2); // initializes all elements of matrix to 0 
	createATrans( L_x-2, dx, dt, A); 

	// Start the iterative solver
	gsl_vector* u_new    = gsl_vector_calloc(L_x-2);
	gsl_vector* u_interm = gsl_vector_calloc(L_x-2);  // u_interm intermediate u

	for (int j=0; j < Titers; j++) {
		CrankNicolsonIteration( A, DiagonalEntries, OffDiagonalEntries, u0, u_new, u_interm);
	}

	// Testing against exact solution
	double ExactSolution = 0.0;
	double sum = 0.0;
	for (int i=0; i<(L_x-2); ++i) {
		ExactSolution = sin( M_PI/l_x*dx*i)*exp( (-1.0)*( M_PI/l_x)*(M_PI/l_x)*Titers*dt);
		sum += fabs((gsl_vector_get(u0,i) - ExactSolution));
	}
	
	std::cout << std::setprecision(5) << std::setiosflags(std::ios::scientific);
	std::cout << "Crank-Nicolson method: L2Error is " << sum/L_x << 
					" in " << Titers << " iterations " << std::endl; 
	
	
	gsl_vector_free(u0);
	gsl_vector_free(u_new);
	gsl_vector_free(u_interm);
	gsl_vector_free(DiagonalEntries);
	gsl_vector_free(OffDiagonalEntries);
	gsl_matrix_free(A);
	
}

void createATrans(int MatrixSIZE, double dx, double dt, gsl_matrix *A ) {
	double alpha_const = dt/(dx*dx);
	
	for (int i = 1; i < MatrixSIZE-1; i++) {
		gsl_matrix_set( A, i, i-1, alpha_const/2.0 ) ; 
		gsl_matrix_set( A, i, i+1, alpha_const/2.0 ) ; 
		gsl_matrix_set( A, i, i, 1.0 - alpha_const ); } 
	
	// set how transformation A deals with "the simple" boundary condition (0 at both ends)
	gsl_matrix_set( A, 0, 0, 1.0-alpha_const) ; 
	gsl_matrix_set( A, 0, 1, alpha_const/2.0) ; 
	gsl_matrix_set( A, MatrixSIZE-1, MatrixSIZE-2, 1.0-alpha_const) ; 
	gsl_matrix_set( A, MatrixSIZE-1, MatrixSIZE-1, alpha_const/2.0) ; 
		
}	

void CrankNicolsonIteration( gsl_matrix *A, gsl_vector *DiagonalEntries, gsl_vector *OffDiagonalEntries,
							gsl_vector *u, gsl_vector *u_new, gsl_vector *u_interm) {
		gsl_blas_dgemv( CblasNoTrans, 1.0, A, u, 0.0, u_interm) ; 
		gsl_linalg_solve_symm_tridiag( DiagonalEntries, OffDiagonalEntries, u_interm, u_new);
		gsl_vector_swap(u,u_new);
}					
									
								
