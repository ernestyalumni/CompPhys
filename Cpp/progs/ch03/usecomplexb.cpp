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
// Program to calculate addition and multiplication of 2 complex numbers
// Hjorth-Jensen says the standard template library (STL) for complex numbers is used in practice
using namespace std;
#include <iostream>
#include <cmath>
#include <complex>
int main()
{
  complex<double> x(6.1,8.2), y(0.5,1.3);
  // write out x+y
  cout << x + y << x*y << endl;
  cout << "Re(x) = " << x.real() << "Im(x) = " << x.imag() << endl;
  // http://en.cppreference.com/w/cpp/numeric/complex
  complex<double> z = x;
  cout << z << endl;
  z += x;
  cout << z << endl;
  z -= y;
  cout << z << endl;
  z /= x;
  cout << z << endl;
  z *= y;
  cout << z << endl;
  cout << "exp(z) = " << exp(z) << endl;
  z = (complex<float>) z;
  cout << "sin(z) = " << sin(z) << endl;
  
  //  cout << "asin(z) = " << asin(z) << endl;
  // /usr/include/c++/5.3.1/cmath:125:3: note:   no known conversion for argument 1 from ‘std::complex<double>’ to ‘float’

  cout << "sinh(z) = " << sinh(z) << endl; 
  cout << "x = " << x << endl;
  cout << "y = " << y << endl;
  return 0;
}
