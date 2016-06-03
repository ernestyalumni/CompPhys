/* 
 * cf. M. Hjorth-Jensen, Computational Physics, University of Oslo (2015) 
 * http://www.mn.uio.no/fysikk/english/people/aca/mhjensen/ 
 * Ernest Yeung (I typed it up and made modifications)
 * ernestyalumni@gmail.com
 * MIT License
 * cf. Chapter 2 Introduction to C++ and Fortran 
 * 2.1.1 Scientific hello world
 * http://folk.uio.no/mhjensen/compphys/programs/chapter02/cpp/program1.cpp
 */
/* comments in C begin like this and end with */
#include <stdlib.h> /* atof function */
#include <math.h>   /* sine function */ 
#include <stdio.h>  /* printf function */

int main (int argc, char* argv[])
{
  double r, s;       /* declare variables */
  r = atof(argv[1]); /* convert the text argv[1] to double */
  s = sin(r);
  printf("Hello, World! sin(%g)=%g=\n", r,s);
  return 0;          /* success execution of the program   */
}


