/* 
 * cf. M. Hjorth-Jensen, Computational Physics, University of Oslo (2015) 
 * http://www.mn.uio.no/fysikk/english/people/aca/mhjensen/ 
 * Ernest Yeung (I typed it up and made modifications)
 * ernestyalumni@gmail.com
 * MIT License
 * cf. Chapter 2 Introduction to C++ and Fortran 
 * 2.5.2 Pointers and arrays in C++
 * http://folk.uio.no/mhjensen/compphys/programs/chapter02/cpp/program7.cpp
 */
using namespace std;
#include <stdio.h>
main()
{
  int var;
  int *pointer;

  pointer = &var;
  var = 421;
  printf("Address of the integer variable var : %p\n", &var);
  printf("Value of var : %d\n", var);
  printf("Value of the integer pointer variable: %p\n", pointer);
  printf("Value which pointer is pointing at : %d\n", *pointer);
  printf("Address of the pointer variable : %p\n", &pointer);
}


