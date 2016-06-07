/* 
 * cf. M. Hjorth-Jensen, Computational Physics, University of Oslo (2015) 
 * http://www.mn.uio.no/fysikk/english/people/aca/mhjensen/ 
 * Ernest Yeung (I typed it up and made modifications)
 * ernestyalumni@gmail.com
 * MIT License
 * cf. Chapter 5 Numerical Integration
 * 5.1 Newton-Cotes Quadrature
 */

/*
 cf. http://stackoverflow.com/questions/1653958/why-are-ifndef-and-define-used-in-c-header-files

#ifndef, #define, #endif are include guards.
Once header is included, it checks if a unique value (in this case HEADERFILE_H or in this particular case, integrate1d_H) is defined.  Then, if it's not defined, it defines it, and continues to the rest of the page.
Also, see https://en.wikipedia.org/wiki/Include_guard
#ifndef test returns false if another file, child.c, has #include but effectively twice,
and so preprocessor skips down to #endif
*/
#ifndef integrate1d_H
#define integrate1d_H

#include <new>
#include <cstdio>
#include <cstdlib>
#include <cmath>

using namespace std;

#define ZERO 1.0E-10

// Function declarations/prototypes

void gauss_legendre(double, double, double *, double *, int);
double trapezoid_rule(double,double,int,double (*func)(double));
double rectangle_rule(double, double, int, double (*func)(double));
double Simpson_rule(double, double, int, double (*func)(double));


#endif
