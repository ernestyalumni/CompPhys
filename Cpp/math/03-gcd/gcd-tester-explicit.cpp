/**
 * 	@file 	gcd-tester-explicit.cpp
 * 	@brief 	A program to test the gcd explicit procedure.  Calculate the greatest common divisor of 2 integers.   
 * 	@ref	Edward Scheinerman.  C++ for Mathematicians: An Introduction for Students and Professionals. 2006.  Ch. 3
 * 
 * COMPILATION TIP : -Wall warning all, -g debugger, gdb can step through 
 * gcc -Wall -g gcd-explicit.cpp gcd-tester-explicit.cpp -o gcd-tester-explicit
 * */
#include "gcd.h" // gcd(long,long)
#include <iostream>

int main() {
	long a,b;
	
	std::cout << "Enter the first number --> "; 
	std::cin >> a;
	std::cout << "Enter the second number --> "; 
	std::cin >> b;
	
	std::cout << "The gcd of " << a << " and " << b << " is " << gcd(a,b) << std::endl; 
	return 0;
}  
