//------------------------------------------------------------------------------
/// \file   factorial.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  A program to test the gcd procedure.  Calculate the greatest common
/// divisor of 2 integers.   
/// \ref    Edward Scheinerman.  C++ for Mathematicians: An Introduction for
/// Students and Professionals. 2006. Ch. 3. Exercise 3.2.
///\details 3 basic elements of recursion: 1. a test to stop or continue
/// recursion, 2. end, base case, 3. a recursion call
/// COMPILATION TIP : -Wall warning all, -g debugger, gdb can step through 
/// g++ -std=c++17 -Wall -g factorial.cpp -o factorial
//------------------------------------------------------------------------------
#include <iostream> // std::cout in main  

long factorial(long n) {
	// base case
	if (n==0) { return 1; }  

	// if n<0 case
	if (n<0) { return n*factorial(n+1); } 
	
	std::cout << "used non-constexpr function of factorial" << '\n';

	// if n>0 case
	return n*factorial(n-1);
}

/**
 * check the assembly generated from this 
 * */
long factorial_fast(long n, long result=1) {
	// base case
	if (n==0) { return result; }
	// this else if statement wasn't included by rapid coder 
	else if (n<0) {
		return factorial_fast(n+1, n*result); 
	}
	else {
		return factorial_fast(n-1, n*result);
	}
}

//------------------------------------------------------------------------------
/// \details C++11 constexpr functions use recursion rather than iteration.
/// (C++14 constexpr functions may use local variables and loops)
/// \ref https://en.cppreference.com/w/cpp/language/constexpr
//------------------------------------------------------------------------------
constexpr int factorial(int n)
{
	return n <= 1 ? 1 : (n * factorial(n - 1));
}

//------------------------------------------------------------------------------
/// \details output function that requires a compile-time constant, for testing
/// \ref https://en.cppreference.com/w/cpp/language/constexpr
//------------------------------------------------------------------------------
template <int n>
struct constN
{
	constN() { std::cout << n << '\n'; }
};

int main() {
	long n;
	std::cout << "Enter the number to compute factorial of --> "; 
	std::cin >> n;
	
	std::cout << "The factorial of " << n << " is " << factorial(n) << std::endl; 

	std::cout << "Using another way (check the assembly generated from it with gdb when you have time) : " << std::endl; 

	std::cout << "The factorial of " << n << " is " << factorial_fast(n) << std::endl; 

	std::cout << "4! = ";
	constN<factorial(4)> out1; // computed at compile time

	return 0;	
}
