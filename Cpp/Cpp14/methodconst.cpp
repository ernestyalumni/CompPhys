/**
 * 	@file 	methodconst.cpp
 * 	@brief 	Meaning of "const" last in a C++ method declaration?   
 * 	@ref	https://stackoverflow.com/questions/751681/meaning-of-const-last-in-a-c-method-declaration 
 * 	@details When you add const keyword to a method, the this pointer will essentially become const, and 
 * therefore you can't change any member data (unless use mutable)
 * 
 * https://stackoverflow.com/questions/11604190/meaning-after-variable-type  
 * "&" meaning after variable type, means you're passing the variable by reference, 
 * The & means function accepts address (or reference) to a variable, instead of value of the variable.  
 * 
 * COMPILATION TIP : -Wall warning all, -g debugger, gdb can step through 
 * g++ -Wall -g methodconst.cpp -o methodconst
 * */
#include <iostream>  

class MyClass
{
	private:
		int counter;
	public:
		void Foo() {
			std::cout << "Foo" << std::endl;
		}
		
		void Foo() const {
			std::cout << "Foo const " << std::endl; 
		}

		/** 
		 * @fn Foo1()
		 * @details non-const method can change instance members, which you can't do in the const version 
		 * */
		void Foo1()
		{
			counter++; // this works
			std::cout << "Foo" << std::endl; 
		}
		
		void Foo1() const 
		{
//			counter++;	// error: increment of member ‘MyClass::counter’ in read-only object
			std::cout << "Foo const" << std::endl; 
		}

};

/**
 * @class MyClass1 
 * @details mark member as mutable and const method can then change it.  Mostly used for internal counters
 * */
class MyClass1
{
	private:
		mutable int counter;
	public:
		MyClass1() : counter(0) {}
		
		void Foo() {
			counter++;
			std::cout << "Foo" << std::endl;
		}
		
		void Foo() const {
			counter++;
			std::cout << "Foo const" << std::endl;
		}
		
		int GetInvocations() const
		{
			return counter;
		}
};


int main() {
	MyClass cc; 
	const MyClass &ccc = cc; 
	cc.Foo(); 
	ccc.Foo();  

	/* my code */
	MyClass * cc_ptr; 
	cc_ptr->Foo(); // Foo 
	const MyClass * ccc_ptr; 
	ccc_ptr->Foo(); // Foo const  

	cc.Foo1(); 	// Foo

	MyClass1 cc1;  
	const MyClass1& ccc1 = cc1;
	cc1.Foo();	// Foo
	ccc1.Foo();	// Foo const 
	std::cout << "The MyClass1 instance has been invoked " << ccc1.GetInvocations() << " times" << std::endl; 

	MyClass1 cc2; 
	cc2.Foo(); 	// Foo
	MyClass1& c2 = cc2;  
	c2.Foo(); // Foo 
	std::cout << "The MyClass1 instance has been invoked " << cc2.GetInvocations() << " times" << std::endl; 

	const MyClass1 ccc2 = cc2;
	ccc2.Foo();  // Foo const 
	std::cout << "The MyClass1 instance has been invoked " << ccc2.GetInvocations() << " times" << std::endl; 


}
		
