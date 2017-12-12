/**
 * 	@file 	gcd.h
 * 	@brief 	Calculate the greatest common divisor of 2 integers.   
 * 	@ref	Edward Scheinerman.  C++ for Mathematicians: An Introduction for Students and Professionals. 2006.  Ch. 3. 
 * 	Program 3.1: The header file gcd.h 
 * 	@details Note: gcd(0,0) will return 0 and print an error message.  
 * 	This header file also illustrates, pedagogically, overloading and pass by reference (pp. 46-47)
 * we can pass values to a procedure (function) in its list of arguments, but procedure can't change value of the 
 * arguments, known as call by value 
 * 	2 different procedures may have same name, but procedures must have different types of arguments.  
 * 	overloading - ability to name different procedures with same name	
 * instead of passing value of x,y to procedure, we pass a reference to x,y
 * instead of sending a copy of x,y to this procedure, send the variable itself (and not a clone) 
 * 
 * 	@param a the first integer 
 * 	@param b the second integer
 * 	@param x,y 
 * 	@return the greatest common divisor of a and b, d  
 * 
 * COMPILATION TIP : -Wall warning all, -g debugger, gdb can step through 
 * gcc -Wall -g gcd.cpp -o gcd
 * */
/* lines #ifndef GCD_H, instruction to preprocessor, stands for "if not defined"  
 * if GCD_H not defined, we should do what follows up to matching #endif
 * #define GCD_H, defines symbol GCD_H, although doesn't specify any particular value 
 * for symbol (we just want to know whether it's defined)  
 * #endif prevent double inclusion  
 * */
#ifndef GCD_H
#define GCD_H  

#include <iostream>

long gcd(long a, long b);

/** @fn gcd
 * 	@param a
 * 	@param b
 * 	@param x
 * 	@param y
 * 	@details Call by reference is mechanism needed so gcd procedure can deliver 3 answers: 
 * 	gcd returned via a return statement, and values x, y
 * */
long gcd(long a, long b, long &x, long &y); 

#endif // END of GCD_H
