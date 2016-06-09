/* argcargv.cpp
 * Ernest Yeung 
 * ernestyalumni@gmail.com
 * I wanted to get a handle on argc argv and so this small script plays around with that
 * dasblinkenlight had a great answer on stackexchange, and I implement his/her suggestion in this script:
 * http://stackoverflow.com/questions/31015912/using-pointers-to-iterate-through-argv
 * Through this exercise, I learned that argc >= 1, as argv always contains at least the name of the program.
 */
using namespace std;
#include <iostream>
#include <cstdlib> /* atoi atof */
#include <typeinfo> /* typeid */

int main (int argc, char* argv[])
{
	cout << "This is your argc : " << argc << endl;
	if (argc == 0 ) {
		cout << "I tested that argc is 0 so I don't think argv contains anything.  I'll loop through it anyways. " << endl;
		for (char **a = argv; a != argv+argc ; a++ ) {
			for (char *p = *a; *p != '\0' ; p++) {
				cout << "This is what argc is containing : " << *p << endl;
			}
		}
	}
	else {
		cout << "I tested that argc is greater than 0.  Here's your argc: " << argc << endl;
		cout << "I'll loop through argv" << endl;
		for (char **a = argv; a != argv+argc ; a++) {
			for (char *p = *a; *p != '\0'; p++) {
				cout << "This is what argc is containing : " << *p << endl;
				cout << "This is the type : " << typeid(*p).name() << endl;
			}
		}
		if (argc > 1) {
			cout << "\n I tested that argc is greater than 1, so argv has more than just the program name " << endl;
			for (int i = 0; i < argc; i++ ){
				cout << "For index " << i << "  this is argv[i] : " << argv[i] << endl;
			}
		}	
	}
}
	
