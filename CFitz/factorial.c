/* factorial.c */
/*
  Fitzpatrick.  Computational Physics.  329.pdf
  2.8 Control Statements
*/

/*
  Program to evaluate factorial of non-negative
  integer n supplied by user.
*/

#include <stdio.h>
#include <stdlib.h>

int main()
{
  int n, count;
  double fact = 1.;

  /* Read in value of n */
  printf("\nn = ");
  scanf("%d", &n);

  /* Reject negative value of n */
  if (n < 0)
    {
      printf("\nError: factorial of negative integer not defined\n");
      exit(1);
    }

  /* Calculate factorial */
  for (count = n; count > 0; --count) fact *= (double) count;

  /* Output result */

  printf("\nn = %5d   Factorial(n) = %12.3e\n", n, fact);

  return 0;
}

