/* 
 * cf. M. Hjorth-Jensen, Computational Physics, University of Oslo (2015) 
 * http://www.mn.uio.no/fysikk/english/people/aca/mhjensen/ 
 * Ernest Yeung (I typed it up and made modifications)
 * ernestyalumni@gmail.com
 * MIT License
 * cf. Chapter 3 Numerical differentiation and interpolation
 * 3.3 Classes in C++
 * 3.3.1 The Complex class
 */
/* 
   "However, writing a complex class yourself is a good pedagogical exercise." - Hjorth-Jensen

   Define first header file complex.h which contains declarations of the class.  
   The header file contains
   - class declaration (data and functions)
   - declaration of stand-alone functions, and all inlined functions, 

*/
/*
  Credit this website:
  http://bhattaca.github.io/cse2122/code/complex_h.html
  for explaining clearly, lucidly, with full working code for complex numbers as an example for 
  C++ classes and header file
 */

#ifndef Complex_H
#define Complex_H
// various include statements and definitions
/* 
   Hjorth-Jensen has include statements, but bhattaca doesn't in his/her header files (???)
#include <iostream>
#include <new>
#include <cmath>
*/

class Complex
{
  // private:
  // double re, im;
 public:
  double r;
  double i;

  Complex(double, double);
  Complex(double );
  Complex();

  Complex add(Complex&);

  Complex operator+ (const Complex&) const;
  Complex operator- ( Complex&) ;
  Complex operator* ( Complex&) ;
  Complex operator/ ( Complex&) ;
  Complex operator/ ( double) ;

  Complex conj();
  double norm();
  void print();

  double Re();
  double Im();
};
// declarations of various functions used by the class

#endif


