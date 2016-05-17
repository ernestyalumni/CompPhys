/* iteration.c */
/*
  Fitzpatrick.  Computational Physics.  329.pdf
  2.8 Control Statements

  EY : 20160517 Compiling note:

  cf. http://stackoverflow.com/questions/5248919/c-undefined-reference-to-sqrt-or-other-mathematical-functions
   Answer from paxdiablo

"You may find that you have to link with the math libraries on whatever system you're using, something like:

gcc -o myprog myprog.c -L/path/to/libs -lm
                                       ^^^ - this bit here."
   So what I did was, on Fedora 23 Linux

   gcc iteration.c -lm
   
   and it worked

   Programming note:
   stdlib.h header has function exit
   function call exit(1) causes program to abort with an error status

   while and do-while statements are particularly well suited to looping situations 
   in which number of passes through loop is NOT known in advance
				    
*/

/* iteration.c */
/*
  Program to solve algebraic equation

   5      2
  x  + a x  - b = 0  

  by iteration. Easily shown that equation must have at least 
  one real root.  Coefficients a and b are supplied by user.

  Iteration scheme is as follows:

                  2  0.2
  x    = ( b - a x  )
   n+1            n

   where x_n is nth iteration.  User must supply initial guess for x.
   Iteration continues until relative change in x per iteration is 
   less than eps (user supplied) or until number of iterations exceeds
   NITER.  Program aborts if (b- a x*x) becomes negative.
*/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>

/* Set max. allowable no. of iterations */
#define NITER 30

int main()
{
  double a, b, eps, x, x0, dx = 1., d;
  int count = 0;

  /* Read input data */
  printf("\na = ");
  scanf("%lf", &a);
  printf("b = ");
  scanf("%lf", &b);
  printf("eps = ");
  scanf("%lf", &eps);

  /* Read initial guess for x */
  printf("\nInitial guess for x = ");
  scanf("%lf", &x);
  x0 = x;

  while (dx > eps) // Start iteration loop: test for convergence
    {
      /* Check for too many iterations */
      ++count;
      if (count > NITER)
	{
	  printf("\nError: no convergence\n");
	  exit(1);
	}

      /* Reject complex roots */
      d = b - a * x * x;
      if (d < 0.)
	{
	  printf("Error: complex roots - try another initial guess\n");
	  exit(1);
	}

      /* Perform iteration */
      x = pow(d, 0.2);
      dx = fabs( (x - x0) / x );
      x0 = x;

      /* Output data on iteration */
      printf("Iter = %3d  x  = %8.4f   dx = %12.3e\n", count, x, dx);
    }
  return 0;
}

    
 
