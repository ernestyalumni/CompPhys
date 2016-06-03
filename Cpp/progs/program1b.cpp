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
// A comment line begins like this in C++ programs
// using namespace std; // in Mac OSX, I obtain this error: program1.cpp:12:17: warning: using directive refers to implicitly-defined namespace 'std'
using namespace std;
#include <iostream>
#include <cstdlib>
#include <cmath>
int main (int argc, char* argv[])
{
  // Read in output file, abort if there are too few command-line arguments
  if (argc <= 1) {
    cout << "Bad Usage: " << argv[0] <<
      " read also a number on the same line, e.g., prog.exe 0.2" << endl;
    exit(1); // here the program stops.
  }
  // convert the text argv[1] to double using atof
  double r = atof(argv[1]);
  double s = sin(r);
  cout << "Hello World (from C++) sin(" << r << ")=" << s << endl; 
  // success
  return 0;
}



