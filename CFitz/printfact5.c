/* printfact5.c */
/*
  Fitzpatrick.  Computational Physics.  329.pdf
  Arrays

*/

/*
  Program to print factorials of all integers
  between 0 and 20
*/

#include <stdio.h>


/* Function prototype for factorial() */
void factorial(double []);

int main()
{
  int j;
  double fact[21];            // Declaration of array fact[]


  /* Calculate factorials */
  factorial(fact);

  /* Output results */
  for (j = 0; j <= 20; ++j)
      printf("j = %3d    factorial(j) = %12.3e\n", j, fact[j]);


  return 0;
}
  
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

void factorial(double fact[])
{
  /*
    Function to calculate factorials of all integers
    between 0 and 20 (in form of floating-point 
    numbers) via recursion formula

    (n+1)! = (n+1) n!

    Factorials returned in array fact[0..20]
  */

  int count;

  fact[0] = 1.;               // Set 0! = 1 

  /* Calculate 1! through 20! via recursion */
  for (count = 0; count < 20; ++count)
    fact[count+1] = (double)(count+1) * fact[count];

  return;

}

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





