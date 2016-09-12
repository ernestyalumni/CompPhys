/* timingcode.cpp
 * Ernest Yeung 
 * ernestyalumni@gmail.com
 * timing code with chrono in C++11/C++14
 * */
// cf. https://solarianprogrammer.com/2012/10/14/cpp-11-timing-code-performance/
// C++11 timing code performance

#include <iostream>
#include <chrono>

using namespace std;

int main() {

	// cf. http://en.cppreference.com/w/cpp/chrono/system_clock
	cout << "system_clock" << endl;
	cout << chrono::system_clock::period::num << endl;
	cout << chrono::system_clock::period::den << endl;
	cout << "steady = " << boolalpha << chrono::system_clock::is_steady << endl << endl;
	
	cout << "high_resolution_clock" << endl;
	cout << chrono::high_resolution_clock::period::num << endl;
	cout << chrono::high_resolution_clock::period::den << endl;
	cout << "steady = " << boolalpha << chrono::high_resolution_clock::is_steady << endl << endl;

	cout << "steady_clock" << endl;
	cout << chrono::steady_clock::period::num << endl;
	cout << chrono::steady_clock::period::den << endl;
	cout << "steady = " << boolalpha << chrono::steady_clock::is_steady << endl << endl;
	
	return 0;
	
}
// For measuring execution time of a piece of code, use the now() function and drop this code in:

/* auto start = chrono::steady_clock::now();
 * 
 * // 
 * // Insert the code that will be timed
 * //
 * 
 * auto end = chrono::steady_clock::now();
 * 
 * // Store the time difference between start and end
 * auto diff = end - start;
 * 
 * // If you want to print the time difference between start and end in the above code, use:
 * 
 * cout << chrono::duration <double, milli> (diff).count() << " ms" << endl;
 * 
 * If you prefer to use nanoseconds, use
 * 
 * cout << chrono::duration <double, nano> (diff).count() << " ns" << endl;
 * 
 * */
