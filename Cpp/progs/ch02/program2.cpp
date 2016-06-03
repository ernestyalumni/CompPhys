/* 
 * cf. M. Hjorth-Jensen, Computational Physics, University of Oslo (2015) 
 * http://www.mn.uio.no/fysikk/english/people/aca/mhjensen/ 
 * Ernest Yeung (I typed it up and made modifications)
 * ernestyalumni@gmail.com
 * MIT License
 * cf. Chapter 2 Introduction to C++ and Fortran 
 * 2.2 Representation of Integer Numbers
 * http://folk.uio.no/mhjensen/compphys/programs/chapter02/cpp/program2.cpp
 */

using namespace std;
#include <iostream>
#include <cstdlib>

int main (int argc, char* argv[])
{
  if (argc <= 1) {
    cout << "Bad Usage: " << argv[0] <<
      " read also a number on the same line, e.g., prog.exe 0.2" << endl;
    exit(1); // here the program stops.
  }


  int i;
  int terms[32]; // storage of a0, a1, etc, up to 32 bits
  int number = atoi(argv[1]);
  // initialize the term a0, a1 etc
  for (i=0; i < 32; i++){ terms[i] = 0; }
  for (i=0; i < 32; i++){
    terms[i] = number%2;
    number /= 2;
  }

  // write out results
  cout << " Number of bytes used= " << sizeof(number) << endl;
  for (i=0; i < 32; i++){
    cout << " Term nr: " << i << " Value= " << terms[i];
    cout << endl;
  }
  return 0;
}


