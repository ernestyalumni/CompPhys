/* quadratic.c
   Fitzpatrick.  Computational Physics.  329.pdf
   2.7 Structure of a C Program

   EY : 20160517 Compiling note:

   cf. http://stackoverflow.com/questions/5248919/c-undefined-reference-to-sqrt-or-other-mathematical-functions
   Answer from paxdiablo

"You may find that you have to link with the math libraries on whatever system you're using, something like:

gcc -o myprog myprog.c -L/path/to/libs -lm
                                       ^^^ - this bit here."
   So what I did was, on Fedora 23 Linux

   gcc quadratic -lm
   
   and it worked
				    
*/

/* quadratic.c */
/*
  Program to evaluate real roots of quadratic equation

     2
  a x  + b x + c = 0 

  using quadratic formula

                     2
  x = ( -b +/- sqrt(b - 4 a c) ) / (2 a)
*/

#include <stdio.h>
#include <math.h>

int main()
{
  double a, b, c, d, x1, x2;

  /* Read input data */

  printf("\na = ");
  scanf("%lf", &a);
  printf("b = ");
  scanf("%lf", &b);
  printf("c = ");
  scanf("%lf", &c);

  /* Perform calculation */
  d = sqrt(b * b - 4. * a * c);
  x1 = (-b + d) / (2. * a);
  x2 = (-b - d) / (2. * a);

  /* Display output */
  printf("\nx1 = %12.3e   x2 = %12.3e\n", x1, x2);

  return 0;
}

