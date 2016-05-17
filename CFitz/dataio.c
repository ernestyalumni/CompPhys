/* dataio.c
   Fitzpatrick.  Computational Physics.  329.pdf
   2.6 Data Input and Output
*/

#include <stdio.h>

int k = 3;
double x = 5.4, y = -9.81;

// When working with a data file, 1st step is to establish buffer area,
// where information is temporarily stored whilst being transferred between
// program and file
// FILE is special structure type that establishes buffer area
// output is identifier (name?) of created buffer area

FILE *output;


int main()
{
  output = fopen("data.out", "w");
  if (output == NULL)
    {
      printf("Error opening file data.out\n");
    }

  fprintf(output, "k = %3d  x + y = %9.4f  x*y = %11.3e\n", k, x + y, x*y);

  fclose(output);
}

