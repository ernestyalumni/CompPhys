/* 
 * cf. M. Hjorth-Jensen, Computational Physics, University of Oslo (2015) 
 * http://www.mn.uio.no/fysikk/english/people/aca/mhjensen/ 
 * Ernest Yeung (I typed it up and made modifications)
 * ernestyalumni@gmail.com
 * MIT License
 * cf. Chapter 3 Numerical differentiation and interpolation
 * 3.2 Numerical Interpolation and Extrapolation
 * 3.2.1 Interpolation
 */
/*
** The function
**        polint()
** takes as input xa[0,..,n-1] and ya[0,..,n-1] together with a given value 
** of x and returns a value y and an error estimate dy.  If P(x) is a polynomial
** of degree N - 1 such that P(xa_i) = ya_i, i = 0,..,n-1, then the returned
** value is y = P(x).
*/
using namespace std;
#include <cmath>
#include <stdio.h> /* printf, fopen  */
#include <stdlib.h> /* exit, EXIT_FAILURE */

const float ZERO = 0.0 ;

void polint(double xa[], double ya[], int n, double x, double *y, double *dy)
{
  int i, m, ns = 1;
  double den, dif, dift, ho, hp, w;
  double *c, *d;

  dif = fabs(x-xa[0]);
  c = new double [n];
  d = new double [n];
  for (i=0; i < n; i++) {
    if ((dift = fabs(x-xa[i])) < dif) {
      ns = i;
      dif = dift;
    }
    c[i] = ya[i];
    d[i] = ya[i];
  }
  *y = ya[ns--];
  for (m=0; m < (n-1); m++) {
    for (i=0; i < n - m; i++) {
      ho = xa[i] - x;
      hp = xa[i+m] - x;
      w = c[i+1] - d[i];
      if ((den = ho - hp) < ZERO) {
	printf("\n\n Error in function polint(): ");
	printf("\nden = ho - hp = %4.1E -- too small\n", den);
	exit(1);
      }
      den = w/den;
      d[i] = hp * den;
      c[i] = ho * den;
    }
    *y += (*dy = (2 * ns < (n - m) ? c[ns + 1] : d[ns--]));
  }
  delete [] d;
  delete [] c;
} // End: function polint()

int main()
{
  return 0;
}
  
