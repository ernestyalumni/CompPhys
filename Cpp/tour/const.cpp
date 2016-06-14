/*
 * const.cpp
 * cf. Bjarne Stroustrup, A Tour of C++, Addison-Wesley Professional (2013)
 * Chapter 1 The Basics
 * 1.7 Constants pp. 8
 * Ernest Yeung  ernestyalumni@gmail.com
 * 20160613
 * Compiling tip: this worked for me if you obtain a c++11 error:
 * g++ -std=c++11 const.cpp
*/
using namespace std;
#include <iostream>

/* const - used to specify interfaces, so data can be passed to functions without fear of modification */

const int dmv = 17; // dmv is a named constant
int var = 17;       // var is not a constant

/* constexpr 
 * roughly means "to be evaluated at compile time."
 * used to specify constant, 
 * allow placement of data in read-only memory (where it's unlikely to be corrupted)
 * and performance */

/* For function to be usable in a constant expression,
 * it must be defined constexpr */
constexpr double square(double x){ return x*x; }
constexpr double max1 = 1.4*square(dmv); // OK if square(17) is a constant expression
const double max3 = 1.4*square(var); // OK, may be evaluated at run time

int main() {
	cout << "This is dmv : " << dmv << endl;
	cout << "This is var : " << var << endl;
	cout << "This is max1 : " << max1 << endl;
	cout << "This is max3 : " << max3 << endl;
	cout << "This is square :" << square << endl;
}
