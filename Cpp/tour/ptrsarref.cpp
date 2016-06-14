/*
 * ptrsarref.cpp
 * cf. Bjarne Stroustrup, A Tour of C++, Addison-Wesley Professional (2013)
 * Chapter 1 The Basics
 * 1.8 Pointers, Arrays, and References pp. 9
 * Ernest Yeung  ernestyalumni@gmail.com
 * 20160613
 * Compiling tip: this worked for me if you obtain a c++11 error:
 * g++ -std=c++11 ptrsarref.cpp
*/
using namespace std;
#include <iostream>

char v[6];
char* p;
char* p6 = &v[3]; // p6 points to v's 4th element
char x = *p6; // *p6 is the object that p6 points to 

void copy_fct()
{
	int v1[10] = {0,1,2,3,4,5,6,7,8,9};
	int v2[10]; // to become a copy of v1
	
	for (auto i=0; i!=10; ++i) // copy elements
		v2[i] = v1[i];
		cout << " This is v1 : " << v1 << " This is v2 : " << v2 << endl; 
}

// C++11 offers a simpler for statement, called range-for-statement
void print()
{
	int v[] = {0,1,2,3,4,5,6,7,8,9};
	
	for (auto x: v) // for each x in v
		cout << x << '\n';
	
	for (auto x: {10,21,32,43,54,65})
		cout << x << '\n';
}

// if we didn't want to copy values from v into the variable x, 
// but rather have x refer to an element,
void increment()
{
	int v[] = {0,1,2,3,4,5,6,7,8,9};
	for (auto& x : v ) {
		++x;
		cout << x << endl;
	}
}

// When we don't want to modify an argument, but still don't want the cost of copying,
// use const

// check a pointer argument actually points to something
int count_x(char* p, char x)
	// count the number of occurrences of x in p[]
	// p is assumed to point to a zero-terminated array of char (or to nothing)
{
	if (p==nullptr) return 0;
	int count = 0 ;
	// note how 
	// we can move a pointer to point to next element of array using ++ and 
	// leave out initializer in for statement if we don't need it
	for (;p!=nullptr; ++p)
		if (*p == x)
			++count;
	return count;
}

int count_x2(char* p, char x)
	// count the number of occurrences of x in p[]
	// p assumed to point to a zero-terminated array of char (or to nothing)
{
	int count = 0;
	while (p) {
		if (*p ==x)
			++count;
		++p;
	}
	return count;
}

int main() {
	for (auto i = 0; i<6; ++i)
		v[i] = 't';
//	v = "testi";
	cout << "This is x: " << x << endl;
	print(); 
	increment();
//	auto result1 = count_x(&v,'t');
//	auto result2 = count_x(&v,'t');
//	cout << "This is the result of count_x : " << result1	<< endl;
//	cout << "This is the result of count_x2 : " << results2 << endl;
}	


		
