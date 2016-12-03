// diffusion1dimplicit.cpp
/* Implicit method for 1-dim. diffusion equation, i.e. backward Euler.
 * While based on 10.2.1 Explicit Scheme of Hjorth-Jensen (2015), see my 
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
#include <gsl/gsl_linalg.h> // gsl_linalg_solve_symm_tridiag

void implicitIteration(gsl_vector *, gsl_vector *, gsl_vector *, gsl_vector * );

int main(int argc, char * argv[]) {
	// physical constants
	double l_x = 1.0;
	
	// discretization parameters
	int L_x = 1000; 
	double dx = l_x / ( static_cast<double>( L_x ));
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
	gsl_vector* DiagonalEntries = gsl_vector_calloc(L_x-2);
	gsl_vector* OffDiagonalEntries = gsl_vector_calloc(L_x-3);
	
	for (int i=0; i<(L_x-2); ++i) {
		gsl_vector_set(DiagonalEntries,i, 1.0 + 2.0*CFL_const ) ;
	}	
	for (int i=0; i<(L_x-3); ++i) {
		gsl_vector_set(OffDiagonalEntries,i, (-1.0)*CFL_const ) ;
	}	

	// Start the iterative solver
	gsl_vector* u_new = gsl_vector_calloc(L_x-2);

	for (int j=0; j < Titers; j++) {
		implicitIteration(DiagonalEntries,OffDiagonalEntries,u0,u_new) ;
	}
	
	// Testing against exact solution
	double ExactSolution = 0.0;
	double sum = 0.0;
	for (int i=0; i<(L_x-2); ++i) {
		ExactSolution = sin( M_PI/l_x*dx*i)*exp( (-1.0)*( M_PI/l_x)*(M_PI/l_x)*Titers*dt);
		sum += fabs((gsl_vector_get(u0,i) - ExactSolution));
	}
	
	std::cout << std::setprecision(5) << std::setiosflags(std::ios::scientific);
	std::cout << "Implicit (tridiagonal solver) method: L2Error is " << sum/L_x << 
					" in " << Titers << " iterations " << std::endl;
	
	gsl_vector_free(u0);
	gsl_vector_free(u_new);
	gsl_vector_free(DiagonalEntries);
	gsl_vector_free(OffDiagonalEntries);
	
}											

/*		
// tridiagSolverprep - preparation for tridiagonal systems solver		
void tridiagSolverprep( double dx, double dt, 
						gsl_vector *DiagonalEntries, gsl_vector *OffDiagonalEntries ) {
	} */
void implicitIteration( gsl_vector *DiagonalEntries, gsl_vector *OffDiagonalEntries,
						gsl_vector *u, gsl_vector *u_new) {
	gsl_linalg_solve_symm_tridiag( DiagonalEntries, OffDiagonalEntries, u, u_new) ;
	gsl_vector_swap(u, u_new);
		
}
