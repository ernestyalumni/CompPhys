/* ftoc.c */
/*
  Fitzpatrick.  Computational Physics.  329.pdf
  2.15 Command Line Parameters pp. 78
  
  EY suffix means, this is my own changes, to explore what char *argv[] means

  "
  The main() function may optionally possess special arguments which allow parameters
  to be passed to this function from the operating system.  There are 2 such arguments,
  convetionally called argc and argv.
  
  argc is an integer which is set to the number of parameters passed to main()
  
  argv is an array of pointers to character strings which contain these parameters

  In order to pass one or more parameters to a C program when it is executed from the operating system, 
  parameters must follow the program name on the command line: e.g.
  
  % program-name parameter_1 parameter_2 parameter_3 .. parameter_n

  Program name will be stored in the first item in argv, followed by each of the parameters. 
  Hence, if program name is followed by n parameters there'll be n + 1 entries in argv,
  ranging from argv[0] to argv[n]
  Furthermore, argc will be automatically set equal to n+1
*/

/*
  Program to convert temperature in Fahrenheit input
  on command line to temperature in Celsius
*/

#include <stdlib.h>
#include <stdio.h>

int main(int argc, char *argv[])
{
  double deg_f, deg_c;

  /* If no parameter passed to program print error
     message and exit */
  if (argc < 2)
    {
      printf("Usage: ftoc temperature\n");
      exit(1);
    }

  /* Convert first command line parameter to double */
  deg_f = atof(argv[1]);
  /* Convert from Fahrenheit to Celsius */
  deg_c = (5. /9. ) * (deg_f - 32.);

  printf("%f degrees Fahrenheit equals %f degrees Celsius\n",
	 deg_f, deg_c);

  return 0;
}


    
	
