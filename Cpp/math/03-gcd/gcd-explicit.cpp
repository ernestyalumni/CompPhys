/**
 * 	@file 	gcd-explicit.cpp
 * 	@brief 	Calculate the greatest common divisor of 2 integers.   Steps shown explicitly.  
 * 	@ref	Edward Scheinerman.  C++ for Mathematicians: An Introduction for Students and Professionals. 2006.  Ch. 3
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
 
 /** 	@fn gcd
  * 	@details arguments are COPIED to a and b, we say C++ procedures CALL BY VALUE, 
  * 		arguments are copies of originals
  */ 
long gcd(long a, long b) {
	 
	 // if a and b are both zero, print an error and return 0 
	if ( (a==0) && (b==0) ) {

	/* object std::cerr similar to std::cout object; wouldn't have been mistake to use std::cout but
	 * std::cerr standard for error messages * */
		 std::cerr << "WARNING: gcd called with both arguments equal to zero."  << std::endl; 
	}
	 
	 // Make sure a and b are both nonnegative
	if (a<0) {
		a = -a;
	}
	if (b<0) {
		b= -b;	
	}
	
	// if a is zero, the answer is b
	if (a==0) {
		return b;
	}
	
	// otherwise, we check all possibilities from 1 to a 
	
	long d; 	// d will hold the answer
	// though d is uninitialized, never fear, by MATH, 1 divides into ALL integers
	
	for (long t=1; t<=a; t++) {
		if ( (a%t==0) && (b%t==0) ) {
			d = t;
		}
		std::cout << " t : " << t << " d : " << d << std::endl;
	}
	
	return d;
}
