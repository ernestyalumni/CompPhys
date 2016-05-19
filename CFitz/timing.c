/* timing.c */
/*
  Fitzpatrick.  Computational Physics.  329.pdf
  Timing 

  EY : 20160517 Compiling note:

  cf. http://stackoverflow.com/questions/5248919/c-undefined-reference-to-sqrt-or-other-mathematical-functions
   Answer from paxdiablo

"You may find that you have to link with the math libraries on whatever system you're using, something like:

gcc -o myprog myprog.c -L/path/to/libs -lm
                                       ^^^ - this bit here."
   So what I did was, on Fedora 23 Linux

   gcc iteration.c -lm
   
   and it worked

  
*/

/*
  Program to test operation of clock() function
*/

#include <stdio.h>
#include <time.h>
#include <math.h>
#define N_LOOP 1000000

int main()
{
  int i;
  double a = 11234567890123456.0, b;
  clock_t time_1, time_2;

  time_1 = clock();
  for (i=0; i < N_LOOP; i++) b = a * a * a * a;
  time_2 = clock();
  printf("CPU time needed to evaluate a*a*a*a:   %f microsecs\n",
	 (double) (time_2 - time_1) / (double) CLOCKS_PER_SEC);

  time_1 = clock();
  for (i=0; i < N_LOOP; i++) b = pow(a, 4.);
  time_2 = clock();
  printf("CPU time needed to evaluate pow(a, 4.): %f microsecs\n",
	 (double) (time_2 - time_1) / (double) CLOCKS_PER_SEC);

  return 0;
}
