/**
 * 	@file 	gcd-extended.cpp
 * 	@brief 	Find the probability that 2 integers in (1,...,n) are relatively prime  
 * 	@ref	Edward Scheinerman.  C++ for Mathematicians: An Introduction for Students and Professionals. 2006.  Ch. 3
 * 	Program 3.11 Code for the extended gcd procedure 
 * 	@details overloading, ability to name different procedures with the same name 
 * 	
 * 	@param a the first integer 
 * 	@param b the second integer
 *  @param x, y such that d = ax + by for d = gcd(a,b)
 * 	@return the greatest common divisor of a and b  
 * 
 * COMPILATION TIP : -Wall warning all, -g debugger, gdb can step through 
 * gcc -Wall -g gcd.cpp -o gcd
 * */
#include "gcd.h"

long gcd(long a, long b, long &x, long &y) { 
	long d; // place to hold final gcd 
	
	// in case b = 0, we have d = |a|, x = 1 or -1, y arbitrary (say, 0) 
	if (b==0) {
		if (a<0) {
			d = -a;
			x = -1;
			y = 0; 
		}
		else {
			d = a;
			x = 1; 
			y = 0;
		}
		return d; 
	}
	
	// if b is negative, here is a workaround
	if (b<0) {
		d = gcd(a,-b,x,y);
		y = -y;
		return d;
	}
	
	// if a is negative, here is a workaround
	if (a<0) {
		d = gcd(-a,b,x,y);
		x = -x;
		return d;
	}
	
	// set up recursion
	long aa = b;
	long bb = a%b; 
	long qq = a/b;
	long xx,yy;
	d = gcd(aa,bb,xx,yy);
	
	x = yy;
	y = xx - qq *yy;
	return d;
}


// gcd(a,b), chosen from before 

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
