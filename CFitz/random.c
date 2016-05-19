/* random.c */
/*
  Program to test operation of rand() function
*/

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#define N_MAX 10000000

int main()
{
  int i, seed;
  double sum_0, sum_1, mean, var, x;

  /* Seed random number generator */
  seed = time(NULL);
  srand(seed);

  /* Calculate mean and variance of x: random number uniformly
     distributed in range 0 to 1 */
  for (i = 1, sum_0 = 0., sum_1 = 0.; i <= N_MAX; i++)
    {
      x = (double) rand() / (double) RAND_MAX;

      sum_0 += x;
      sum_1 += (x - 0.5) * (x - 0.5);
    }
  mean = sum_0 / (double) N_MAX;
  var = sum_1 / (double) N_MAX;

  printf("mean(x) = %12.10f  var(x) = %12.10f\n", mean, var);

  return 0;
}

