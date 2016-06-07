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
/* 
** Big note - at least to me, program1.cpp in pp. 51 of Hjorth-Jensen's notes
** code appears incomplete, and wouldn't compile (I don't think).  I looked here:
** https://github.com/CompPhysics/ComputationalPhysicsMSU/blob/master/doc/Programs/LecturePrograms/programs/Classes/cpp/program1.cpp
** and the entire code for program1.cpp appears there
*/

using namespace std;
// #include <iostream> I don't think this is correct of Hjorth-Jensen since printf,scanf are used
#include <stdio.h>
#include <cmath>

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

//   Read in from screen the air temp, the number of steps
//   final time and the initial temperature */

void initialize(double *initial_step, double *x, int *number_of_steps)
{
  printf("Read in from screen initial step, x, and number of steps\n");
  scanf("%lf %lf %d", initial_step, x, number_of_steps);
  return;
}  // end of function initialize

// This function computes the second derivative

void second_derivative( int number_of_steps, double x, double initial_step,
			double *h_step, double *computed_derivative)
{
  int counter;
  double h;

  //    calculate the step size
  //    initialize the derivative, y and x (in minutes)
  //    and iteration counter

  h = initial_step;

  // start computing for different step sizes
  for (counter=0; counter < number_of_steps; counter++ )
    {

      // setup arrays with derivatives and step sizes

      h_step[counter] = h;
      computed_derivative[counter] =
	(exp(x+h)-2.*exp(x) + exp(x-h))/(h*h);
      h = h*0.5;
    } // end of do loop

  return;
  
} // end of function second derivative


// function to write out the final results
void output(double *h_step, double *computed_derivative, double x,
	    int number_of_steps)
{
  int i;
  FILE *output_file;
  output_file = fopen("out.dat", "w");
  for (i=0; i < number_of_steps; i++)
    {
      fprintf(output_file, "%12.5E %12.5E \n",
	      log10(h_step[i]),log10(fabs(computed_derivative[i]-exp(x))/exp(x)));
    }
  fclose(output_file);

} // end of function output

