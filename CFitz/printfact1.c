/* printfact1.c */
/*
  Fitzpatrick.  Computational Physics.  329.pdf
  2.9 Functions

  Programming note from Fitzpatrick.  pp.50
  "Ideally, function definitions should always precede the corresponding function
  calls in a C program.  

  Unfortunately, for the sake of logical clarity, most C programmers prefer to place
  the main() function at the beginning of their programs.

  ...compilation errors can be avoided by using a construct known
  as a function prototype"
*/

/*
  Program to print factorials of all integers
  between 0 and 20
*/

#include <stdio.h>
#include <stdlib.h>

/* Prototype for function factorial() */
double factorial(int);

int main()
{
  int j;

  /* Print factorials of all integers between 0 and 20 */
  for (j = 0; j <= 20; ++j)
    printf("j = %3d    factorial(j) = %12.3e\n", j, factorial(j));

  return 0;
}

  

  
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

double factorial(int n)
{
  /*
    Function to evaluate factorial (in floating-point form)
    of non-negative integer n.
  */

  int count;
  double fact = 1.;

  /* Abort if n is negative integer */
  if (n < 0)
    {
      printf("\nError: factorial of negative integer not defined\n");
      exit(1);
    }

  /* Calculate factorial */
  for (count = n; count > 0; --count) fact *= (double) count;

  /* Return value of factorial */
  return fact;
}

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  

