/* randomEY.c */
/*
  Program to test operation of rand() function

  I took random.c and made changes to play around it
*/

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#define N_MAX 10000000

// Function prototypes
int randommeanvar(int);

int main()
{
  int i, seed;
  double sum_0, sum_1, mean, var, x;

  /* Seed random number generator */
  seed = time(NULL);

  printf("\nThis is seed's value (which is time(NULL)), in decimal, string, and float, resp.:\n");
  //   printf("%d %s %12.10f", seed, seed, seed); // Segmentation fault (core dumped)
  printf("%d \n", seed);

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

  printf("\nN=10:\n");
  randommeanvar(10);

  //  printf("\nN=100:\n");
  //  randommeanvar(100);

  //  printf("\nN=1000:\n");
  //  randommeanvar(1000);

  //  printf("\nN=1000000000:\n");
  //  randommeanvar(1000000000);

  
  return 0;
}

int randommeanvar(int N) {
  
  double sum_0, sum_1, mean, var, x;
  for (int i = 1, sum_0 = 0., sum_1 = 0.; i <= N; i++)
    {
      x = (double) rand() / (double) RAND_MAX;
      printf("\nx: %f\n",x);
      
      sum_0 += x;
      sum_1 += (x - 0.5) * (x - 0.5);

      printf("\nsum_0, sum_1: %f %f \n",sum_0,sum_1);
    }
  mean = sum_0 / (double) N;
  var  = sum_1 / (double) N;

  printf("mean(x) = %12.10f  var(x) = %12.10f\n", mean, var);

  return 0;
}


