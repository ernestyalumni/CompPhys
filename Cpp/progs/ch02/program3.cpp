/* 
 * cf. M. Hjorth-Jensen, Computational Physics, University of Oslo (2015) 
 * http://www.mn.uio.no/fysikk/english/people/aca/mhjensen/ 
 * Ernest Yeung (I typed it up and made modifications)
 * ernestyalumni@gmail.com
 * MIT License
 * cf. Chapter 2 Introduction to C++ and Fortran 
 * 2.2 Representation of Integer Numbers
 * http://folk.uio.no/mhjensen/compphys/programs/chapter02/cpp/program3.cpp
 */
// Program to calculate 2**n
using namespace std;
#include <iostream>
#include <cmath>

int main()
{
  int int1, int2, int3;
  // print to screen
  cout << "Read in the exponential N for 2^N =\n";
  // read from screen
  cin >> int2;
  int1 = (int) pow(2., (double) int2);
  cout << " 2^N * 2^N = " << int1*int1 << "\n";
  int3 = int1 - 1;
  cout << " 2^N*(2^N - 1) = " << int1 * int3 << "\n";
  cout << " 2^N - 1 = " << int3 << "\n";
  return 0;
}
// End: program main()
 
