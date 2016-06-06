/* 
 * cf. M. Hjorth-Jensen, Computational Physics, University of Oslo (2015) 
 * http://www.mn.uio.no/fysikk/english/people/aca/mhjensen/ 
 * Ernest Yeung (I typed it up and made modifications)
 * ernestyalumni@gmail.com
 * MIT License
 * cf. Chapter 3 Numerical differentiation and interpolation
 * 3.1 Numerical Differentiation
 * 3.1.1.1 Initializations and main program
 */
/*
**   Program to compute the second derivative of exp(x).
**   Three calling functions are included
**   in this version.  In one function we read in the data from screen, 
**   the next function computes the second derivative
**   while the last function prints out data to screen.
*/
using namespace std;
#include <iostream>

void initialize(double *, double *, int *);
void second_derivative(int, double, double, double *, double *);
void output(double *, double *, double, int);

int main()
{
  // declarations of variables
  int number_of_steps;
  double x, initial_step;
  double *h_step, *computed_derivative;
  // read in input data from screen
  initialize(&initial_step, &x, &number_of_steps);
  // allocate space in memory for the one-dimensional arrays
  // h_step and computed_derivative
  h_step = new double[number_of_steps];
  computed_derivative = new double[number_of_steps];
  // compute the second derivative of exp(x)
  second_derivative(number_of_steps, x, initial_step, h_step, computed_derivative);
  // Then we print the results to file
  output(h_step, computed_derivative, x, number_of_steps);
  // free memory
  delete [] h_step;
  delete [] computed_derivative;
  return 0;
} // end main program
						 
