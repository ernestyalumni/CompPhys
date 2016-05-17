/* printfact3.c */
/*
  Fitzpatrick.  Computational Physics.  329.pdf
  2.10 Pointers

*/

/*
  Program to print factorials of all integers
  between 0 and 20
*/

#include <stdio.h>
#include <stdlib.h>

/* Prototype for function factorial() */
void factorial(int, double *);

int main()
{
  int j;
  double fact;
  
  /* Print factorials of all integers between 0 and 20 */
  for (j = 0; j <= 20; ++j)
    {
      factorial(j, &fact);
      printf("j = %3d    factorial(j) = %12.3e\n", j, fact);
    }      
  return 0;
}
  
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

void factorial(int n, double *fact)
{
  /*
    Function to evaluate factorial *fact (in floating-point form)
    of non-negative integer n.
  */

  *fact = 1.;

  /* Abort if n is negative integer */
  if (n < 0)
    {
      printf("\nError: factorial of negative integer not defined\n");
      exit(1);
    }

  /* Calculate factorial */
  for (; n > 0; --n) *fact *= (double) n;

  /* Return value of factorial */
  return;
}

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



