/*
 * struct.cpp
 * cf. Bjarne Stroustrup, A Tour of C++, Addison-Wesley Professional (2013)
 * Chapter 2 User-Defined Types
 * 2.2 Structures pp. 16
 * Ernest Yeung  ernestyalumni@gmail.com
 * 20160613
 * Compiling tip: this worked for me if you obtain a c++11 error:
 * g++ -std=c++11 ptrsarref.cpp
*/
using namespace std;
#include <iostream>

// struct
struct Vector {
	int sz;
	double* elem;
};

/* & in Vector& indicates that we pass v by non-const reference (Sec. 1.8);
 * that way, vector_init() can modify the vector passed to it */
/* new operator allocates memory from dynamic memory i.e. free store */
void vector_init(Vector& v, int s)
{
	v.elem = new double[s]; // allocate an array of s doubles
	v.sz = s;
}

double read_and_sum(int s)
// read s integers from cin and return their sum; s is assumed to be positive
{ 
	Vector v;
	vector_init(v,s); // allocate s elements for v
	for (int i=0; i!=s; ++i)
		cin >> v.elem[i]; // read into elements
		
	double sum  =0;
	for (int i=0; i!=s; ++i)
		sum+=v.elem[i]; // take the sum of the elements
	return sum;
}

void f(Vector v, Vector& rv, Vector* pv)
{
	int i1 = v.sz; // access through name
	int i2 = rv.sz; // access through reference
	int i4 = pv->sz; // access through pointer

	cout << "This is i1 : " << i1 << endl;
	cout << "This is i2 : " << i2 << endl;
	cout << "This is i4 : " << i4 << endl;
}
 
// 2.3 Classes
class Vector2{
	public:
		Vector2(int s) :elem{new double[s]}, sz{s} {} // construct a Vector
		double& operator[](int i) { return elem[i]; } // element access: subscripting
		int size() { return sz; }
	private:
		double* elem; // pointer to the elements
		int sz;       // the number of elements
};

double read_and_sum2(int s)
{
	Vector2 v(s);  // make a vector of s elements
	for (int i=0; i!=v.size(); ++i )
		cin >> v[i];  // read into elements
		
	double sum = 0;
	for (int i=0; i!=v.size(); ++i)
		sum +=v[i];  // take the sum of the elements
	return sum;
}
/* notice member initializer list :elem{new double[s]}, sz{s}
 * initialize elem with pointer to s elements of type double
 * */


Vector2 v(6);	
		/*
// 2.5 Enumerations

enum class Color { red, blue, green };
enum class Traffic_light { green, yellow, red };

Color col = 
*/

int main() {
	cout << "Practice using structs." << endl;
	Vector v1;
	vector_init( v1 ,3);
	cout << "This is v1.sz : " << v1.sz << endl;
	cout << "This is v1.elem : " << v1.elem << endl;
	cout << "This is v1.elem[0] : " << v1.elem[0] << endl;
	cout << "This is v1.elem[1] : " << v1.elem[1] << endl;
	cout << "I will start read_and_sum for 3 elements. \n";
	double sumeg = read_and_sum(3);
	cout << "This is the result of read_and_sum : " << sumeg << endl;

	Vector v, rv, pv;
	vector_init( v,3);
	vector_init( rv,4);
	vector_init( pv, 5);
	
	Vector* ppv = &pv;
	f(v,rv,ppv); 
	
	cout << "Now, enter 3 numbers for the class version of Vector, Vector2 : " << endl;
	double sumresult = read_and_sum2(3);
	cout << "This is the result of read_and_sum2 : " << sumresult << endl;
}
