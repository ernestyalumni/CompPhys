/* average.c */
/* 
   Program to calculate the average of n numbers and then 
   compute the deviation of each number from the average

   Code adapted from "Programming with C", 2nd Edition, Byron Gottfreid,
   Schaum's Outline Series, (McGraw-Hill, New York NY, 1996)
*/

#include <stdio.h>
#include <stdlib.h>

#define NMAX 100

int main()
{
  int n, count;
  double avg, d, sum = 0.;
  double list[NMAX];
  
  /* Read in value for n */
  printf("\nHow many numbers do you want to average? ");
  scanf("%d", &n);

  /* Check that n is not too large or too small */
  if ((n > NMAX) || (n <= 0))
    {
      printf("\nError: invalid value for n\n");
      exit(1);
    }

    /* Read in the numbers and calculate their sum */
  for (count = 0; count < n; ++count)
    {
      printf("i = %d  x = ", count + 1);
      scanf("%lf", &list[count]);
      sum += list[count];
    }

    /* Calculate and display the average */
  avg = sum / (double) n;
  printf("\nThe average is %5.2f\n\n", avg);
    
    /* Calculate and display the deviations about the average */
  for (count = 0; count < n; ++count)
    {
      d = list[count] - avg;
      printf("i = %d  x = %5.2f  d = %5.2f\n", count + 1, list[count], d);
    }
  return 0;
}

