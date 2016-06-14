/*
 * test.cpp
 * cf. Bjarne Stroustrup, A Tour of C++, Addison-Wesley Professional (2013)
 * Chapter 1 The Basics
 * 1.9 Tests pp. 12
 * Ernest Yeung  ernestyalumni@gmail.com
 * 20160613
 * Compiling tip: this worked for me if you obtain a c++11 error:
 * g++ -std=c++11 test.cpp
*/
using namespace std; // makes names from std visible without std::
#include <iostream> // cout, cin

bool accept()
{
	cout << "Do you want to proceed (y or n)?\n"; // write question
	
	char answer = 0;
	cin >> answer;
	
	if (answer == 'y')
		return true;
	return false;
}

/* switch tests a value against a set of constants.  */
bool accept2()
{
	cout << "Do you want to proceed (y or n)?\n"; // write question
	
	char answer = 0;
	cin >> answer;
	
	switch (answer) {
		case 'y':
			return true;
		case 'n':
			return false;
		default:
			cout << "I'll take that for a no.\n";
			return false;
	}
}
/*
void action()
{
	while(true) {
		cout << "enter action:\n" // request action
		string act;
		cin >> act; // rear characters into a string
		Point delta {0,0}; // Point holds an {x,y} pair
		
		for (char ch: act) {
			switch (ch) {
				case 'u': // up
				case 'n': // north
					++delta.y;
					break;
				case 'r': // right
				case 'e': // east
					++delta.x;
					break;
				default:
					cout << "I freeze!\n";
			}
		}
	}
}
*/

void action()
{
	while(true) {
		cout << "enter action:\n" << endl; // request action
		string act;
		cin >> act; // rear characters into a string
		int delta[] = {0,0}; // Point holds an {x,y} pair
		
		for (char ch: act) {
			switch (ch) {
				case 'u': // up
					cout << 'u' << endl;
				case 'n': // north
					++delta[1];
					break;
				case 'r': // right
				case 'e': // east
					++delta[0];
					break;
				default:
					cout << "I freeze!\n";
			}
			break;
		}
	}
}

int main()
{
	bool acceptflag = true;
	acceptflag = accept();
	cout << "Result of accept function: " << acceptflag << endl; 
	bool accept2flag = true;
	accept2flag = accept2();
	cout << "Result of accept2 function: " << accept2flag << endl;	

	cout << "Time to run action " << endl;
	action();
}
	
