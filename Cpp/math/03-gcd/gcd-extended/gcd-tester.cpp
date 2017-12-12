/**
 * 	@file 	gcd-tester.cpp
 * 	@brief 	A program to test the gcd procedure.  Calculate the greatest common divisor of 2 integers.   
 * 	@ref	Edward Scheinerman.  C++ for Mathematicians: An Introduction for Students and Professionals. 2006.  Ch. 3
 * 	Program 3.6: A program to test the gcd procedure.   
 * 
 * COMPILATION TIP : -Wall warning all, -g debugger, gdb can step through 
 * gcc -Wall -g gcd.cpp gcd-tester.cpp -o gcd-tester
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
