/**
 * 	@file 	gcd-recursive.cpp
 * 	@brief 	Calculate the greatest common divisor of 2 integers, recursively.   
 * 	@ref	Edward Scheinerman.  C++ for Mathematicians: An Introduction for Students and Professionals. 2006.  Ch. 3
 * 	Program 3.3: Beginning of the file gcd.cpp 
 * 	@details Strategy: test successive integers to see if they're divisors of a and b, 
 * 	keep track of largest value that divides both.  
 * 	
 * 	@param a the first integer 
 * 	@param b the second integer
 * 	@return the greatest common divisor of a and b  
 * 
 * COMPILATION TIP : -Wall warning all, -g debugger, gdb can step through 
 * gcc -Wall -g gcd.cpp -o gcd
 * */
#include "gcd.h"
#include <iostream>

long gcd(long a, long b) {
	// Make sure a and b are both nonnegative
	if (a<0) { a = -a; }
	if (b<0) { b = -b; }
	
	// if a and b are both zero, print as error and return 0 
	if ( (a==0) && (b==0) ) {
		std::cerr << "WARNING: gcd called with both arguments equal to zero." 
			<< std::endl;
		return 0;
	}
	
	// The following 2 cases are the base cases for recursion
	// If b is zero, the answer is a 
	if (b==0) { return a; }
	
	// If a is zero, the answer is b
	if (a==0) { return b; }
	
	long c = a % b;
	
	return gcd(b,c);
}
	
		
