/* expressions.c
   Fitzpatrick.  Computational Physics.  329.pdf
   2.3 Expressions and Statements
*/

#include <stdio.h>
#include <math.h>

int a = 3, b = 5;
double factor = 1.2E-5;

// symbolic constant is a name that substitutes for a sequence of characters

#define PI 3.141593

int k;
double x,y;

int kk = 3;
double xx = 5.4, yy = -9.81;

int main()
{
  int check_input = scanf("%d %lf %lf", &k, &x, &y);
  if (check_input < 3)
    {
      printf("Error during data input\n");
    }

  printf("%d %f %f\n", kk, xx, yy);

  printf("kk = %3d xx + yy = %9.4f xx*yy = %11.3e\n", k, x + y, x*y);
  
}


  
