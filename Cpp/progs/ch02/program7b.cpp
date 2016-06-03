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
#include <typeinfo>
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
  printf("truth operator : var, 421 -> : %d\n", var==421);
  printf("typeof : 421 -> : %s\n", typeid( typeof(421)).name() );

  printf("typeof : 421.23 -> : %s\n", typeid( typeof(421.32)).name() );
  printf("typeof : 'jaja' -> : %s\n", typeid( typeof("jaja")).name() );
  printf("typeof : 'u' -> : %s\n", typeid( typeof('u')).name() );

  // I'm experimenting here:
  printf("typeof : 421.23 -> : %d\n", typeid( typeof(421.32)).name() ); // 0.000000
  printf("typeof : 'jaja' -> : %d\n", typeid( typeof("jaja")).name() ); // 0.000000
  printf("typeof : 'u' -> : %d\n", typeid( typeof('u')).name() );       // 0.000000

}


