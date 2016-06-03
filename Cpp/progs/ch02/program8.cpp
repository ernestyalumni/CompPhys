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
#include <iostream>
#include <stdio.h>
int main()
{
  int matr[2];
  int *pointer;
  pointer = &matr[0];
  matr[0] = 321;
  matr[1] = 322;
  printf("\nAddress of the matrix element matr[1]: %p", &matr[0]);
  printf("\nValue of the matrix element matr[1]: %d",matr[0]);
  printf("\nAddress of the matrix element matr[2]: %p",&matr[1]);
  printf("\nValue of the matrix element matr[2]: %d\n", matr[1]);
  printf("\nValue of the pointer : %p", pointer);
  printf("\nValue which pointer points at : %d", *pointer);
  printf("\nValue which (pointer+1) points at: %d\n",*(pointer +1));
  printf("\nAddress of the pointer variable: %p\n",&pointer);
}

