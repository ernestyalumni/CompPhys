/* main.c */
/*
  Fitzpatrick.  Computational Physics.  329.pdf
  2.14 Multi-file

  global variable declarations in the file main.c

  Compiling note:
  gcc main.c factorial.c
  and it worked
*/

/*
  Program to print factorials of all integers
  between 0 and 20
*/

#include <stdio.h>

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





