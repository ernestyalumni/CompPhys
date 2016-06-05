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
  Example of program which uses our complex class
*/

#include <iostream> /* cout cin*/
#include <cmath>
using namespace std;

// include the complex numbers class
#include "complex.h"

int main()
{
  /* This was Hjorth-Jensen's original code commented; 
     it doesn't necessary work (out of the box)!
  Complex a(0.1,1.3); // we declare a complex variable a
  Complex b(3.0), c(5.0,-2.3); // we declare complex variables b and c
  Complex d = b; // we declare a new complex variable d
  cout << "d=" << d << ", a=" << a << ", b=" << b << endl;
  d = a*c + b/a; // we add, multiply and divide two complex numbers
  cout << "Re(d)=" << d.Re() << ", Im(d)=" << d.Im() << endl; // write out the real and imaginary parts
  */

  Complex aa(0.1,1.3); // we declare a complex variable aa
  aa.print();

  Complex bb(3.0), cc(5.0,-2.3); // we declare a complex variable bb and cc
  bb.print();
  cc.print();

  Complex dd = bb;
  // cout << "d=" << dd.print() << ", a=" << aa.print() << ", b=" << bb.print() << endl;
  /* You'll obtain this error message if you do the above
     cannot convert ‘dd.Complex::print()’ (type ‘void’) to type ‘const unsigned char*’
   */
  cout << "dd=" << endl;
  dd.print();
  cout << "aa=" << endl;
  aa.print();
  cout << "bb=" << endl;
  bb.print();

  
  dd = bb/aa ; // we add, multiply and divide 2 complex numbers
  Complex tmp = aa*cc ; 
  dd = dd + tmp ; 

  cout << "Re(dd)=" << dd.Re() << ", Im(dd) = " << dd.Im() << endl; // write out the real and imaginary parts
  
  // cf. http://bhattaca.github.io/cse2122/code/complex_main_cpp.html

  // a = 5 + 3i
  Complex a(5,3);
  cout << "Constructor with two values:\na = ";
  a.print();

  // b = 3 - 4i
  Complex b(3,-4);
  cout << "b = ";
  b.print();

  cout << endl;

  Complex c(4);
  cout << "Constructor with one value:\nc = ";
  c.print();

  // c = a.add(b);
  c = a + b;
  cout << "\na + b = ";
  c.print();

  Complex d;
  cout << "\nDefault constructor:\n d = ";
  d.print();

  d = a - b;
  cout << "\na - b = ";
  d.print();

  Complex e = a * b;
  cout << "\na * b = ";
  e.print();

  Complex f = a / b;
  cout << "\na / b = ";
  f.print();

  cout << "\nNorm of b: " << b.norm() << endl;

  Complex h = a.conj();
  cout << "Complex conjugate of a = ";
  h.print();
  
  return 0;
}

