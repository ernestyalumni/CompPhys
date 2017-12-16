/**
 * 	@file 	future1.cpp
 * 	@brief 	C++ program to demonstrate future, future example    
 * 	@ref	http://www.cplusplus.com/reference/future/future/
 * 	@details object that can retrieve value from some provider object or function, properly synchronizing this access if in different threads  
 * 
 * COMPILATION TIP : -Wall warning all, -g debugger, gdb can step through 
 * g++ -Wall -g future.cpp -o future
 * */
#include <iostream> 	// std::cout
#include <future>		// std::async, std::future
#include <chrono>		// std::chrono::milliseconds  

// a non-optimized way of checking for prime numbers:
bool is_prime(int x) {
	for (int i=2; i<x; ++i) {
		if (x % i == 0) 	// check if x,i are relatively prime 
		{
			return false; 
		}
	}
	return true;
}

int main() 
{
	// call function asynchronously:
	std::future<bool> fut = std::async(is_prime,444444443); // undefined reference to `pthread_create'
	
	// do something while waiting for function to set future:
	std::cout << "checking, please wait";
	std::chrono::milliseconds span(100);
	while (fut.wait_for(span)==std::future_status::timeout) {
		std::cout << '.' << std::flush; 
	}
	
	bool x = fut.get();		// retrieve return value 	
		
	std::cout << "\n444444443 " << (x ? "is" : "is not") << " prime.\n"; 
	
	return 0;
}
	
	
