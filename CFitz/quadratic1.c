/* quadratic1.c
   Fitzpatrick.  Computational Physics.  329.pdf
   2.8 Control Statements

   EY : 20160517 Compiling note:

   cf. http://stackoverflow.com/questions/5248919/c-undefined-reference-to-sqrt-or-other-mathematical-functions
   Answer from paxdiablo

"You may find that you have to link with the math libraries on whatever system you're using, something like:

gcc -o myprog myprog.c -L/path/to/libs -lm
                                       ^^^ - this bit here."
   So what I did was, on Fedora 23 Linux

   gcc quadratic -lm
   
   and it worked

   Programming note:
   stdlib.h header has function exit
   function call exit(1) causes program to abort with an error status
				    
*/

/* quadratic1.c */
/*
  Program to evaluate real roots of quadratic equation

     2
  a x  + b x + c = 0 

  using quadratic formula

                     2
  x = ( -b +/- sqrt(b - 4 a c) ) / (2 a)

  Program rejects cases where roots are complex or where a = 0. 
*/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>

int main()
{
  double a, b, c, d, e, x1, x2;

  /* Read input data */

  printf("\na = ");
  scanf("%lf", &a);
  printf("b = ");
  scanf("%lf", &b);
  printf("c = ");
  scanf("%lf", &c);

  /* Perform calculation */
  e = b * b - 4. * a * c;

  if (e < 0.)
    {
      printf("\nError: roots are complex\n");
      exit(1);
    }

  /* Test for a = 0. */
  if (a == 0.)
    {
      printf("\nError: a = 0.\n");
      exit(1);
    }

  /* Perform calculation */
  d = sqrt(e);
  
  x1 = (-b + d) / (2. * a);
  x2 = (-b - d) / (2. * a);

  /* Display output */
  printf("\nx1 = %12.3e   x2 = %12.3e\n", x1, x2);

  return 0;
}

