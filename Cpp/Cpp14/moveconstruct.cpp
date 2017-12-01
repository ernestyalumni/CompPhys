/**
 * 	@file 	moveconstruct.cpp
 * 	@brief 	move constructor example  
 * 	@ref	http://en.cppreference.com/w/cpp/language/move_constructor
 * https://stackoverflow.com/questions/11572669/move-with-vectorpush-back
 * 	@details  
 * 
 * COMPILATION TIP : -Wall warning all, -g debugger, gdb can step through 
 * g++ -Wall -g moveconstruct.cpp -o moveconstruct
 * */
#include <string>		// std::string 
#include <iostream>
#include <iomanip>
#include <utility>  

#include <vector> 

struct A 
{
	std::string s;
	// struct constructor 
	A() : s("test") { }
	// copy constructor
	A(const A& o) : s(o.s) { std::cout << "move failed!\n"; } 
	// move constructor
	A(A&& o) noexcept : s(std::move(o.s)) { }
}; 

A f(A a)
{
	return a;
}

struct B : A
{
	std::string s2;
	int n;
	// implicit move constructor B::(B&&) 
	// calls A's move constructor 
	// calls s2's move constructor
	// and makes a bitwise copy of n 
}; 

struct C : B 
{
	~C() { } 	// destructor prevents implicit move constructor C::(C&&) 
}; 
	
struct D : B 
{
	D() {  }
	~D() {  }			// destructor would prevent implicit move constructor D::(D&&) 
	D(D&&) = default; 	// forces a move constructor anyways 
}; 

struct X { 
	int a;
	int x;
};

struct Y { 
	int a;
	int x;
	// constructor
	Y() : a(5), x(3) {}
	// destructor
	~Y() {}
	// copy constructor
	Y(const Y& y) : a(y.a), x(y.x) { std::cout << "Copying with Y " << std::endl; }
	// move constructor 
	Y(Y&& y) noexcept : a(std::move(y.a)), x(std::move(y.x)) { std::cout << " Moving with Y " << std::endl; }

};	


int main() 
{
	std::cout << "Trying to move A\n"; 
	A a1 = f(A());			// move-constructs from rvalue temporary
	A a2 = std::move(a1); 	// move constructs from xvalue 
	
	std::cout << "Trying to move B\n"; 
	B b1;
	std::cout << "Before move, b1.s = " << std::quoted(b1.s) << "\n";
	B b2 = std::move(b1); // calls implicit move constructor
	std::cout << "After move, b1.s = " << std::quoted(b1.s) << "\n" ; 
	
	std::cout << "Trying to move C\n"; 
	C c1;
	C c2 = std::move(c1); // calls copy constructor 
	
	std::cout << "Trying to move D\n"; 
	D d1; 
	D d2 = std::move(d1); 

	/* cf. https://stackoverflow.com/questions/11572669/move-with-vectorpush-back
	 * using push_back(x) would create a copy of the object, while push_back(move(x)) 
	 * would tell push_back() that it may "steal" the contents of x, leaving x in an unusable and undefined state.  
	 * 
	 * Consider if you had a vector of lists (std::vector<std::list<int> >) 
	 * and you wanted to push a list containing 100,000 elements. Without move(), 
	 * the entire list structure and all 100,000 elements will be copied. With move(), 
	 * some pointers and other small bits of data get shuffled around, and that's about it. 
	 * This will be lots faster, and will require less overall memory consumption.
	 * */
	X x1;
	X x2; 
	std::vector<X> vecX; 
	vecX.push_back(x1);
	vecX.push_back(std::move(x2)); 

	Y y1;
	Y y2; 
	std::vector<Y> vecY; 
	vecY.push_back(y1);  // Copying with Y
	vecY.push_back(std::move(y2));  //  Moving with Y; Moving with Y 

}
