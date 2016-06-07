/* 
 * cf. M. Hjorth-Jensen, Computational Physics, University of Oslo (2015) 
 * http://www.mn.uio.no/fysikk/english/people/aca/mhjensen/ 
 * Ernest Yeung (I typed it up and made modifications)
 * ernestyalumni@gmail.com
 * MIT License
 * cf. Chapter 5 Numerical Integration
 * 5.1 Newton-Cotes Quadrature
 */

double trapezoidal_rule(double a, double b, int n, double (*func)(double))
{
  double trapez_sum;
  double fa, fb, x, step;
  int j;
  step = (b-a)/((double) n);
  fa = (*func)(a)/2 ;
  fb = (*func)(b)/2 ;
  TrapezSum = 0. ;
  for (j=1; j <= n-1; j++){
    x = j*step + a;
    trapez_sum+=(*func)(x);
  }
  trapez_sum = (trapez_sum+fb+fa)+step;
  return trapez_sum;
} // end trapezoidal_rule

