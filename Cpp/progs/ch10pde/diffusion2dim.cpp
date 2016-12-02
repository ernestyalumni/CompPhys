// diffusion2dim.cpp
/* Simple program from solving the two-dimensional diffusion
	equation or Poisson equation using Jacobi's iterative method
	Note that this program does not contain a loop over the time
	dependence
*/	
/*
 * diffusion2dim.cpp is based on Hjorth-Jensen's implementation, but
 * he uses armadillo as a linear algebra library; here I'm using gsl
 * the original link: 
 * https://raw.githubusercontent.com/CompPhysics/ComputationalPhysics/master/doc/Programs/LecturePrograms/programs/PDE/cpp/diffusion2dim.cpp
 * 
 * */
/*
 * Compilation tip
 * 
 * g++ -lgsl -lgslcblas diffusion2dim.cpp -o diffusion2dim
 * 
 * */

#include <iostream>
#include <iomanip>  // std::setiosflags
#include <gsl/gsl_matrix.h>  
#include <gsl/gsl_math.h> // M_PI

int JacobiSolver(int, double, double, gsl_matrix *, gsl_matrix *, double);

int main(int argc, char * argv[]) {
	int Npoints = 40; 
	double ExactSolution;
	double dx = 1.0/(Npoints - 1);
	double dt = 0.25 *dx*dx;
	double tolerance = 1.0e-14;
	
	gsl_matrix* A = gsl_matrix_calloc(Npoints, Npoints);  // initializes all elements of matrix to 0 
	gsl_matrix* q = gsl_matrix_calloc(Npoints, Npoints);
	
	
	
	// setting up an additional source term
	for (int i = 0 ; i < Npoints; i++) {
		for (int j = 0; j < Npoints; j++) {
			gsl_matrix_set(q,i,j, -2.0*M_PI*M_PI*sin(M_PI*dx*i)*sin(M_PI*dx*j)) ; }}
			
	int itcount = JacobiSolver(Npoints, dx, dt, A, q, tolerance) ;
	
	// Testing against exact solution
	double sum = 0.0;
	for (int i = 0; i < Npoints; i++) {
		for (int j = 0; j < Npoints; j++) {
			ExactSolution = -sin(M_PI*dx*i)*sin(M_PI*dx*j);
			sum += fabs((gsl_matrix_get(A, i,j) - ExactSolution));
		}
	}
	std::cout << std::setprecision(5) << std::setiosflags(std::ios::scientific);
	std::cout << "Jacobi: L2Error is " << sum/Npoints << " in " << itcount << " iterations" << std::endl;	

	gsl_matrix_free(A);
	gsl_matrix_free(q);

}

// Function for setting up the iterative Jacobi solver
int JacobiSolver(int N, double dx, double dt, gsl_matrix *A, gsl_matrix *q, double abstol) 
{
	int MaxIterations = 100000;
	gsl_matrix* Aold = gsl_matrix_calloc(N,N);
	
	double D = dt/(dx*dx) ;
	
	for (int i = 1; i < N-1; i++) {
		for (int j = 1; j < N-1 ; j++) {
			gsl_matrix_set(Aold,i,j, 1.0); }}
	
	// Boundary Conditions -- all zeros
	for (int i = 0; i < N; i++) {
		gsl_matrix_set(A,0,i,   0.0);
		gsl_matrix_set(A,N-1,i, 0.0);
		gsl_matrix_set(A,i,0,   0.0);
		gsl_matrix_set(A,i,N-1, 0.0);
	}
	
	// Start the iterative solver
	for (int k = 0; k < MaxIterations; k++) {
		for (int i = 1; i < N-1; i++) {
			for (int j=1; j < N-1; j++) {
				gsl_matrix_set(A,i,j, dt* gsl_matrix_get( q,i,j) + gsl_matrix_get(Aold,i,j) + 
					D*(gsl_matrix_get(Aold,i+1,j)+gsl_matrix_get(Aold,i,j+1) - 4.0*gsl_matrix_get(Aold,i,j) +
						gsl_matrix_get(Aold,i-1,j) + gsl_matrix_get(Aold,i,j-1))
						);
			}
		}
		double sum = 0.0;
		for (int i = 0; i < N;i++) {
			for (int j = 0; j < N; j++) {
				sum += (gsl_matrix_get(Aold,i,j)-gsl_matrix_get(A,i,j))*(gsl_matrix_get(Aold,i,j)-gsl_matrix_get(A,i,j));
				gsl_matrix_set(Aold,i,j, gsl_matrix_get(A,i,j) );
			}
		}
		if (sqrt(sum) < abstol) {
			return k;
		}
	}
	std::cerr << "Jacobi: Maximum Number of Iterations Reached without Convergence\n";

	gsl_matrix_free(Aold);
	return MaxIterations;
}	
