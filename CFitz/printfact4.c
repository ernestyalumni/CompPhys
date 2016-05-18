/* printfact4.c */
/*
  Fitzpatrick.  Computational Physics.  329.pdf
  2.11 Global

*/

/*
  Program to print factorials of all integers
  between 0 and 20
*/

/*
  From Fitzpatrick in 2.11
  "Global variables should be used sparingly in scientific programs (or any other type of program), since there are inherent dangers in their employment. An alteration in the value of a global variable within a given function is carried over into all other parts of the program. Unfortunately, such an alteration can sometimes happen inadvertently, as the side-effect of some other action. Thus, there is the possibility of the value of a global variable changing unexpectedly, resulting in a subtle programming error which can be extremely difficult to track down, since the offending line could be located anywhere in the program. Similar errors involving local variables are much easier to debug, since the scope of local variables is far more limited than that of global variables."
*/

#include <stdio.h>
#include <stdlib.h>

/* Prototype for function factorial() */
void factorial();

/* Global variable declarations */
int j;

double fact;

int main()
{
  
  /* Print factorials of all integers between 0 and 20 */
  for (j = 0; j <= 20; ++j)
    {
      factorial();
      printf("j = %3d    factorial(j) = %12.3e\n", j, fact);
    }      
  return 0;
}
  
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

void factorial()
{
  /*
    Function to evaluate factorial *fact (in floating-point form)
    of non-negative integer n.
  */

  int count;

  /* Abort if j is negative integer */
  if (j < 0)
    {
      printf("\nError: factorial of negative integer not defined\n");
      exit(1);
    }

  /* Calculate factorial */
  for (count = j, fact = 1.; count > 0; --count) fact *= (double) count;

  /* Return value of factorial */
  return;
}

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





