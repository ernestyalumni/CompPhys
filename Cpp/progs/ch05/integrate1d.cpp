/* 
 * cf. M. Hjorth-Jensen, Computational Physics, University of Oslo (2015) 
 * http://www.mn.uio.no/fysikk/english/people/aca/mhjensen/ 
 * Ernest Yeung (I typed it up and made modifications)
 * ernestyalumni@gmail.com
 * MIT License
 * cf. Chapter 5 Numerical Integration
 * 5.1 Newton-Cotes Quadrature
 * I also took a look here:
cf. https://github.com/CompPhysics/ComputationalPhysicsMSU/blob/7e3178b1bd27716db5f2d3bf39455ee51ad63110/doc/Programs/LecturePrograms/programs/cppLibrary/lib.cpp
 */
/* 
void gauss_legendre(double x1, double x2, double x[], double w[], int n)
** takes the lower and upper limits of integration x1, x2, calculates
** and return the abcissas in x[0,...,n - 1] and the weights in w[0,...,n - 1]
** of length n of the Gauss-Legendre n-point quadrature formula.  

double rectangle_rule(double a, double b, int n, double (*func)(double))
** integration by rectangle rule, in a and b and number of points n and name
** of function

double trapezoid_rule(double a, double b, int n, double (*func)(double))
** integration by trapezoid rule, in a and b and number of points n and name
** of function
*/

#include "integrate1d.h"

void gauss_legendre(double x1, double x2, double x[], double w[], int n)
{
  int    m,j,i;
  double z1,z,xm,xl,pp,p3,p2,p1;
  double const pi = 3.14159265359;
  double *x_low, *x_high, *w_low, *w_high;

  m  = (n+1)/2;             // roots are symmetric in the interval
  xm = 0.5 * (x2 + x1);
  xl = 0.5 * (x2 - x1);

  x_low  = x;               // pointer initialization
  x_high = x + n - 1; 
  w_low  = w;
  w_high = w + n - 1;

  for (i=1; i <= m; i++) {  // loops over desired roots
    z = cos(pi * (i - 0.25)/(n + 0.5));

    /* 
    ** Starting with the above approximation to the ith root
    ** we enter the main loop of refinement by Newton's method.
    */

    do {
      p1 = 1.0;
      p2 = 0.0;

      /*
      ** loop up recurrence relation to get the 
      ** Legendre polynomial evaluated at x
      */

      for (j = 1; j <= n; j++) {
	p3 = p2;
	p2 = p1;
	p1 = ((2.0 * j - 1.0) * z * p2 - (j - 1.0) * p3)/j;
      }

      /* 
      ** p1 is now the desired Legendre polynomial.  Next compute
      ** ppp, its derivative, by standard relation involving also p2, 
      ** polynomial of one lower order.
      */

      pp = n * (z * p1 - p2)/(z * z - 1.0);
      z1 = z ;
      z = z1 - p1/pp;        // Newton's method
    }
    while (fabs(z - z1) > ZERO);

    /*
    ** Scale the root to the desired interval and put in its symmetric
    ** counterpart.  Compute the weight and its symmetric counterpart
    */

    *(x_low++)  = xm - xl * z;
    *(x_high--) = xm + xl * z;
    *w_low      = 2.0 * xl/((1.0 - z*z) * pp * pp);
    *(w_high--) = *(w_low++);
  }
} // End_ function gauss_legendre()


double trapezoid_rule(double a, double b, int n, double (*func)(double))
{
  double trapez_sum;
  double fa, fb, x, step;
  int    j;
  step = (b-a)/((double) n);
  fa   = (*func)(a)/2. ;
  fb   = (*func)(b)/2. ;
  trapez_sum = 0.;
  for (j=1; j <= n-1; j++) {
    x = j*step + a;
    trapez_sum += (*func)(x);
  }
  trapez_sum = (trapez_sum + fb + fa)*step;
  return trapez_sum;
} // end trapezoidal_rule

double rectangle_rule(double a, double b, int n, double (*func)(double))
{
  double rectangle_sum;
  double fa, fb, x, step;
  int    j;
  step = (b-a)/((double) n);
  rectangle_sum = 0.;
  for (j=0; j <= n; j++) {
    x = (j+0.5)*step + a;            // midpoint of a given rectangle
    rectangle_sum += (*func)(x); // add value of function.
  }
  rectangle_sum *= step;         // multiply with step length.
  return rectangle_sum;
} // end rectangle_rule


double Simpson_rule(double a, double b, int n, double (*func)(double))
{
  double simpson_sum;
  double fa, fb, x, step;
  int    j, m;
  step = (b-a)/((double) n);
  fa   = (*func)(a)/2. ;
  fb   = (*func)(b)/2. ;
  simpson_sum = 0.;
  m = n/2;
  for (j=1; j<= m ; j++) {
    x = a + (2.*j - 1.)*step ;
    simpson_sum += 4. * (*func)( x ) ;
      }
  for (j=1; j< m; j++) {
    x = a + (2.* j)*step ;
    simpson_sum += 2. * (*func)(x) ; 
      }
  simpson_sum = (simpson_sum + fa + fb)*step/3.;
  return simpson_sum;
}
	 
