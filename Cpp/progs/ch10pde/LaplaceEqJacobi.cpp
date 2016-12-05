// LaplaceEqJacobi.cpp
/* Jacobi method for 2-dim. Laplace equation. 
 * While based on 10.3.2 Jacobi Algorithm for solving Laplace’s Equation of 
 * Hjorth-Jensen (2015), and 
 * see my 
 * Cebeci, Shao, Kafyeke, Laurendeau. Computational Fluid Dynamics for Engineers (2005),
 * 4.5.2. Iterative Methods
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

void LaplaceEqJacobi(const int, const int, gsl_matrix *, gsl_matrix *, gsl_matrix *, 
						gsl_vector *, gsl_vector *, const double, const double, const double );

int main(int argc, char * argv[]) {
	// physical constants
	double l_x = 1.0;
	double l_y = 1.0;
	
	// discretization parameters
	int L_x = 1000;
	int L_y = 1000;
	
	double dx = l_x / (static_cast<double>(L_x));
	double dy = l_y / (static_cast<double>(L_y));
	
	// boundary conditions
	gsl_vector* LHS_u = gsl_vector_calloc(L_y); 
	gsl_vector* RHS_u = gsl_vector_calloc(L_y); 
	
	for (int j=0; j<L_y; ++j) {
		gsl_vector_set(LHS_u,j, sin(M_PI*dy*j) );
		gsl_vector_set(RHS_u,j, exp(M_PI)*sin(M_PI*dy*j) ); }

	// setting up the iterative solver
//	double tolerance = 1.0e-14;
	const int MaxIterations = 1000;


	// initial guess for matrix, representing 2-dimensional grid
	gsl_matrix* u    = gsl_matrix_calloc(L_x,L_y); // initializes all elements of matrix to  0
	gsl_matrix* unew = gsl_matrix_calloc(L_x,L_y); // initializes all elements of matrix to  0
	gsl_matrix* rho  = gsl_matrix_calloc(L_x,L_y); // initializes all elements of matrix to  0

	
	// set up coefficients for Laplace Eq. discretization
	double theta_x  = dy*dy/(2.0*(dx*dx+dy*dy)) ;
	double theta_y  = dx*dx/(2.0*(dx*dx+dy*dy)) ;
	double delta_xy = dx*dx*dy*dy/(2.0*(dx*dx+dy*dy));
	
	for (int k = 0; k < MaxIterations; k++) {
		LaplaceEqJacobi( L_x, L_y, u, unew, rho, LHS_u, RHS_u, theta_x, theta_y, delta_xy ) ;  
	}

	// Testing against exact solution
	double ExactSolution = 0.0;
	double sum = 0.0;
	for (int j=0; j < L_y; ++j) {
		for (int i=0; i < L_x; ++i) {
			ExactSolution = exp( M_PI*dx*i)*sin(M_PI*dy*j);
			sum += fabs((gsl_matrix_get(unew,i,j) - ExactSolution));
		}
	}
	
	std::cout << std::setprecision(5) << std::setiosflags(std::ios::scientific);
	std::cout << "Jacobi method: L2Error is " << sum/(L_x*L_y) << " in " <<
		MaxIterations << " iterations " << std::endl;

	gsl_vector_free(LHS_u);
	gsl_vector_free(RHS_u);
	gsl_matrix_free(rho);
	gsl_matrix_free(u);
	gsl_matrix_free(unew);
}

void LaplaceEqJacobi( const int L_x, const int L_y, 
					gsl_matrix *u, gsl_matrix *unew, gsl_matrix *rho,
					gsl_vector *LHS_u, gsl_vector *RHS_u,
					const double theta_x, const double theta_y, const double delta_xy
//					const double tolerance) {
) {
	// grid points adjacent to boundary
	for (int i=0; i < L_x ; ++i) {
		// corner cases
		if (i==0) {
			gsl_matrix_set( unew, 0,0,
				theta_x*(gsl_matrix_get(u,1,0)+gsl_vector_get(LHS_u,0))+
				theta_y*(gsl_matrix_get(u,0,1)+ 0) + delta_xy*4.0*M_PI*gsl_matrix_get(rho,0,0));
			gsl_matrix_set( unew, 0,L_y-1,
				theta_x*(gsl_matrix_get(u,1,L_y-1)+gsl_vector_get(LHS_u,L_y-1))+
				theta_y*(0 + gsl_matrix_get(u,0,L_y-2)) + delta_xy*4.0*M_PI*gsl_matrix_get(rho,0,L_y-1));
		}
		else if (i==(L_x-1)) {
			gsl_matrix_set( unew, L_x-1,0,
				theta_x*(gsl_vector_get(RHS_u,0)+gsl_matrix_get(u,L_x-2,0))+
				theta_y*(gsl_matrix_get(u,L_x-1,1)+0) + delta_xy*4.0*M_PI*gsl_matrix_get(rho,L_x-1,0));
			gsl_matrix_set( unew, L_x-1,L_y-1,
				theta_x*(gsl_vector_get(RHS_u,L_y-1)+gsl_matrix_get(u,L_x-2,L_y-1))+
				theta_y*(0 + gsl_matrix_get(u,L_x-1,L_y-2)) + delta_xy*4.0*M_PI*gsl_matrix_get(rho,L_x-1,L_y-1));
		}
		// END all 4 corner cases
		else {
			gsl_matrix_set(unew,i,0,
				theta_x*(gsl_matrix_get(u,i+1,0) + gsl_matrix_get(u,i-1,0))+
				theta_y*(gsl_matrix_get(u,i,1)+0) + delta_xy*4.0*M_PI*gsl_matrix_get(rho,i,0));
			gsl_matrix_set(unew,i,L_y-1,
				theta_x*(gsl_matrix_get(u,i+1,L_y-1)+gsl_matrix_get(u,i-1,L_y-1))+
				theta_y*(0+gsl_matrix_get(u,i,L_y-2)) + delta_xy*4.0*M_PI*gsl_matrix_get(rho,i,L_y-1));
		
		}
	}
	
	for (int j=1; j < (L_y-1); ++j) {
		gsl_matrix_set( unew,0,j,
			theta_x*(gsl_matrix_get(u,1,j) + gsl_vector_get(LHS_u,j)) + 
			theta_y*(gsl_matrix_get(u,0,j+1) + gsl_matrix_get(u,0,j-1)) + 
				delta_xy*4.0*M_PI*gsl_matrix_get(rho,0,j));
		gsl_matrix_set( unew,L_x-1,j,
			theta_x*(gsl_vector_get(RHS_u,j) + gsl_matrix_get(u,L_x-2,j)) + 
			theta_y*(gsl_matrix_get(u,L_x-1,j+1) + gsl_matrix_get(u,L_x-1,j-1)) + 
				delta_xy*4.0*M_PI*gsl_matrix_get(rho,L_x-1,j));
	}
	
	// EY: I'll update old u near (within the "halo" or "stencil") the boundary; let's see if it'll accelerate convergence
	for (int i=0; i<L_x; ++i) {
		gsl_matrix_set( u,i,0, gsl_matrix_get( unew,i,0));
		gsl_matrix_set( u,i,L_y-1, gsl_matrix_get( unew,i,L_y-1)); }
	for (int j=1; j<(L_y-1); ++j) {
		gsl_matrix_set( u,0, j,gsl_matrix_get( unew,0,j));
		gsl_matrix_set( u,L_x-1,j, gsl_matrix_get( unew,L_x-1,j)); }
		
	// interior grid points at least "halo" or "stencil" away from boundary
	for (int j=1; j<(L_y-1); ++j) {
		for (int i=1; i<(L_x-1); ++i) {
			gsl_matrix_set(unew,i,j,
				theta_x*(gsl_matrix_get(u,i+1,j) + gsl_matrix_get(u,i-1,j)) + 
				theta_y*(gsl_matrix_get(u,i,j+1) + gsl_matrix_get(u,i,j-1)) + 
					delta_xy*4.0*M_PI*gsl_matrix_get(rho,i,j)); 
		}
	}
	gsl_matrix_swap(u,unew);


}	
				

	
