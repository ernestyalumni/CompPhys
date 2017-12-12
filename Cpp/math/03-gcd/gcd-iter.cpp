/**
 * 	@file 	gcd-iter.cpp
 * 	@brief 	Calculate the greatest common divisor of 2 integers, recursively.   
 * 	@ref	Edward Scheinerman.  C++ for Mathematicians: An Introduction for Students and Professionals. 2006.  Ch. 3
 * 	Program 3.8: An iterative procedure for gcd 
 * 	@details Strategy: test successive integers to see if they're divisors of a and b, 
 * 	keep track of largest value that divides both.  
 * 	
 * 	@param a the first integer 
 * 	@param b the second integer
 * 	@return the greatest common divisor of a and b  
 * 
 * COMPILATION TIP : -Wall warning all, -g debugger, gdb can step through 
 * gcc -Wall -g gcd-tester-explicit.cpp gcd-iter.cpp -o gcd-tester-iter
 * */
#include "gcd.h"
#include <iostream>

long gcd(long a, long b) {
	// Make sure a and b are both nonnegative 
	if (a<0) { a = -a; }
	if (b<0) { b = -b; }
	
	// if a and b are both zero, print an error and return 0 
	if ( (a==0) && (b ==0)){
		std::cerr << "WARNING: gcd called with both arguments equal to zero." 
			<< std::endl;
		return 0;
	}
	
	long new_a, new_b; // place to hold new versions of a and b

	/*
	 * We use the fact that gcd(a,b) = gcd(b,c) where c = a%b.  
	 * Note that if b is zero, gcd(a,b) = gcd(a,0) = a.  If a is zero, 
	 * and b is not, we get a%b equal to zero, so new_b will be zero,
	 * hence b will be zero and the loop will exit with a == 0, which
	 * is what we want. 
	 */
	
	while (b != 0) {
		new_a = b; 
		new_b = a%b;  // c = a%b
		
		a = new_a; // gcd(b,c)
		b = new_b; 
	}
	
	return a;
}
