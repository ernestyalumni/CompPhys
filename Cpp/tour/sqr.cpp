/*
 * sqr.cpp
 * cf. Bjarne Stroustrup, A Tour of C++, Addison-Wesley Professional (2013)
 * Chapter 1 The Basics
 * 1.3 Hello, World! pp. 3
 * Ernest Yeung  ernestyalumni@gmail.com
 * 20160613
*/
using namespace std; // makes names from std visible without std::
#include <iostream> // include ("import") the declarations for the I/O stream

double square(double x) // square a double precision floating-point number
{
	return x*x;
}

void print_square(double x)
{
	cout << "the square of " << x << " is " << square(x) << "\n";
}

int main()
{
	print_square(1.234); // print: the square of 1.234 is 1.52276
}
