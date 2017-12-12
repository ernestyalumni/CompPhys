/**
 * 	@file 	exhaust.cpp
 * 	@brief 	USE exhaust1.cpp instead; this script is pedagogical.  Find the probability that 2 integers in (1,...,n) are relatively prime  
 * 	@ref	Edward Scheinerman.  C++ for Mathematicians: An Introduction for Students and Professionals. 2006.  Ch. 3
 * 	Program 3.9: A program to calculate p_n 
 * 	@details 2 nested for loops run through all possible a,b with 1<= a,b<=n
 * 	USE exhaust1.cpp instead because for efficiency, notice gcd(10,15) = gcd(15,10) so we can make
 * 	exhaust.cpp twice as fast by calculating only 1 of these: also, we don't have to bother calculating 
 * 	gcd(a,a)
 * 	
 * 	@param a the first integer 
 * 	@param b the second integer
 * 	@return the greatest common divisor of a and b  
 * 
 * COMPILATION TIP : -Wall warning all, -g debugger, gdb can step through 
 * gcc -Wall -g gcd.cpp -o gcd
 * */
#include <iostream>
#include "gcd.h"

int main() {
	long n;
	std::cout << "Enter n --> ";
	std::cin >> n ;
	
	long count = 0;
	
	// 2 nested for loops run through all possible values of a and b with 1 <= a,b <= n 
	for (long a=1; a<=n; a++) {
		for (long b=1; b<=n; b++) {
			if (gcd(a,b) == 1) {
				count++; 
			}
		}
	}
	
	// At the end, we divide the counter by n^2 to get the probability.  
	std::cout << double(count) / (double(n) * double(n)) << std::endl; 
	return 0;
}
	
