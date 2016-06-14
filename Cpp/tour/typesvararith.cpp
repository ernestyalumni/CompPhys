/*
 * typesvararith.cpp
 * cf. Bjarne Stroustrup, A Tour of C++, Addison-Wesley Professional (2013)
 * Chapter 1 The Basics
 * 1.5 Types, Variables, and Arithmetic pp. 5
 * Ernest Yeung  ernestyalumni@gmail.com
 * 20160613
 * Compiling tip: this worked for me if you obtain a c++11 error:
 * g++ -std=c++11 typesvararith.cpp
*/
using namespace std; // makes names from std visible without std::
#include <iostream> // cout, cin
#include <cmath>

void some_function() // function that doesn't return a value
{
	double d = 2.2;
	int i = 7;
	cout << "This is d before : " << d << endl;
	d = d+i;
	cout << "This is d now : " << d << "\n This is i before : " << i << endl;
	i = d*i;
	cout << "This is i now : " << i << endl;
}

double d1 = 2.3; // initialize d1 to 2.3
double d2 {2.3}; // initialize d2 to 2.3
// you'll obtain this error with double d2 {2.3} 
// typesvararith.cpp:26:11: warning: extended initializer lists only available with -std=c++11 or -std=gnu++11
// see the note on how to compile this above

// complex<double> z = 1;
// vector<int> v {1,2,3,4,5,6};

/* use auto where we don't have a specific reason to mention type explicitly;
 * "specific reasons" not to use auto
 * definition is in large scope where we want type clearly visible to readers
 * want explicit about variable's range or precision (e.g. double rather than float)
 * 
 * Compiling note: auto needs -std=c++11, C++11 otherwise warning is obtained
 * */
 
auto b = true;  // a bool
auto ch = 'x';  // a char
auto i = 123;   // an int
auto d = 1.2;   // a double
double y = 1.42;
auto z = sqrt(y); // z has the type of whatever sqrt(y) returns

int main()
{
	some_function() ; 
	cout << "This is d1 : " << d1 << endl; 
	cout << "This is d2 : " << d2 << endl;
//	cout << "This is z  : " << z << endl;
// 	cout << "This is v  : " << v << endl;
	cout << "This is b : " << b << endl;
	cout << "This is ch : " << ch << endl;
	cout << "This is i : " << i << endl;
	cout << "This is d : " << d << endl;
	cout << "This is z : " << z << endl;
	
}

	
