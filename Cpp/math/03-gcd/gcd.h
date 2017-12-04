/**
 * 	@file 	gcd.h
 * 	@brief 	Calculate the greatest common divisor of 2 integers.   
 * 	@ref	Edward Scheinerman.  C++ for Mathematicians: An Introduction for Students and Professionals. 2006.  Ch. 3. 
 * 	Progra 3.1: The header file gcd.h 
 * 	@details Note: gcd(0,0) will return 0 and print an error message.  
 * 	
 * 	@param a the first integer 
 * 	@param b the second integer
 * 	@return the greatest common divisor of a and b  
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

long gcd(long a, long b);

#endif // END of GCD_H
