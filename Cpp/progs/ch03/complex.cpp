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
  "Next we provide a file complex.cpp where the code and algorithms of different functions (except inlined functions) declared within the class are written." -Hjorth-Jensen
*/

/*
  cf. http://bhattaca.github.io/cse2122/code/complex_cpp.html
  I personally think that it's a disservice that the full code for the complex number class wasn't
  included by Hjorth-Jensen; it really is a disservice to the complete novice that wants to 
  learn rapidly; bhattaca provided a working code in full with lucid explanations and no 
  code typos
 */
#include <iostream>
#include <cmath> /* sqrt pow pow(7.0, 3.0) is 7^3 */
using namespace std;

#include "complex.h"

// Constructor with two arguments
Complex::Complex(double _r, double _i)
{
  r = _r;
  i = _i;
}

// Constructor with one argument
Complex::Complex(double _r)
{
  r = _r;
  i = 0.0;
}

// default constructor
Complex::Complex()
{
  r = 0.0;
  i = 0.0;
}

/*
 * Simple function to add two complex numbers.  To add a and b, we have 
 * to make the following call: a.add(b); This looks a little artificial
 * but can be changed to a more natural a + b by using operator
 * overloading.
 */
Complex Complex::add(Complex &rhs)
{
  // (a + bi) + (c+ di) = (a+c) + (b+d)i

  double a = r;
  double b = i;

  double c = rhs.r;
  double d = rhs.i;
  
  double nr = a + c;
  double ni = b+d;

  Complex tmp(nr, ni);
  return tmp;
}

/* Operator overloading
 *  Normally we can use operators like +, -, * etc only for primitive
 *  data types in C++, like int, char, float, double etc.
 *  However, C++ allows us to extend these operators to work on our
 *  classes by overloading them.  The following function overloads
 *  the '+' operator so that we can use a simple command like c = a + b 
 *  to add two objects of type Complex.
 */
Complex Complex::operator+ (Complex &rhs)
{
  // (a + bi) + (c + di) = (a+c) + (b+d)i
  // double a  = r     ; double b = i;
  // double c  = rhs.r ; double d = rhs.i;
  // double nr = a + c;
  // double ni = b + d;
  // Complex tmp(nr, ni);
  // return tmp;
  return Complex(r + rhs.r, i + rhs.i);
}

// Overloaded '-' operator
Complex Complex::operator- (Complex &rhs)
{
  // (a + bi) - (c + di) = (a-c) + (b-d)i
  // double a = r     ; double b = i;
  // double c = rhs.r ; double d = rhs.i;
  // double nr = a - c;
  // double ni = b - d;
  // Complex tmp(nr, ni);
  // return tmp;
  return Complex(r - rhs.r, i - rhs.i);
}

// Overloaded '*' operator
Complex Complex::operator* (Complex &rhs)
{
  // (a + bi)*(c+di) = (ac-bd) + (ad+dc)i
  // double a  = r            ; double b = i;
  // double c  = rhs.r        ; double d = rhs.i;
  // double nr = (a*c) - (b*d);
  // double ni = (a*d) + (b*c);
  // Complex tmp(nr,ni);
  // return tmp;
  return Complex((r * rhs.r) - (i * rhs.i), (r * rhs.i) + (i * rhs.r));
}

// To be completed in next class

/*
  'this' pointer is a constant pointer that holds the memory address of the current object
  cf. http://www.geeksforgeeks.org/this-pointer-in-c/
  'this' in C++ is just a pointer to the current instance
*/

Complex Complex::operator/ (Complex &rhs)
{
  Complex c = rhs.conj();
  double n  = rhs.norm();
  // Complex tmp = (*this)*c;
  // tmp = tmp / (n*n);
  // return tmp;
  return ((*this) * c)/ pow(n,2.0);
}

Complex Complex::operator/ (double rhs)
{
  return Complex(r/rhs, i/rhs);
}

// returns the conjugate of the complex number for which it is called
Complex Complex::conj()
{
  // conj a + bi = a - bi
  return Complex(r,-i);
}

// returns the norm of the complex number for which it is called
double Complex::norm()
{
  // norm a + bi = square_root(a^2 + b^2)
  return sqrt(r*r + i*i);
}

// prints the complex number
void Complex::print()
{
  if (i<0)
    cout << r << " - " << (-i) << "i" << endl;
  else
    cout << r << " + " << i << "i" << endl;
}

double Complex::Re()
{
  return r;
}

double Complex::Im()
{
  return i;
}

/* Hjorth-Jensen's code - I'm not sure if it works as is or even is the correct thing to put
   into a .cpp file
class Complex
{
private:
  double re, im; // real and imaginary part
public:
  Complex ();                             // Complex c;
  Complex (double re, double im = 0.0);   // Definition of a complex variable;
  Complex (const Complex& c);
  Complex& operator = (const Complex& c); // c = a; // equate two complex variables, same as previous
  -Complex () {}                          // destructor
  double Re () const;                     // double real_part = a.Re();
  double Im () const;                     // double imag_part = a.Im();
  double abs () const;                    // double m = a.abs(); // modulus
  friend Complex operator+ (const Complex& a, const Complex& b);
  friend Complex operator- (const Complex& a, const Complex& b);
  friend Complex operator* (const Complex& a, const Complex& b);
  friend Complex operator/ (const Complex& a, const Complex& b);
};
*/

