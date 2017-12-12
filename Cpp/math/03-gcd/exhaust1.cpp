/**
 * 	@file 	exhaust1.cpp
 * 	@brief 	Find the probability that 2 integers in (1,...,n) are relatively prime  
 * 	@ref	Edward Scheinerman.  C++ for Mathematicians: An Introduction for Students and Professionals. 2006.  Ch. 3
 * 	Program 3.10 A slightly better program to calculate p_n
 * 	@details second for loop begins with b=a+1, so inner loop runs from a+1 up to n.  
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
	std::cin >> n;
	
	long count = 0;
	
	for (long a=1; a<=n; a++) {
		for (long b=a+1; b<=n; b++) {
			if (gcd(a,b) == 1) {
				count++;
			}
		}
	}
	count = 2*count + 1;
	
	std::cout << double(count) / (double(n) * double(n)) << std::endl; 
	return 0; 
}  
