/**
 * 	@file 	lambdaexp.cpp
 * 	@brief 	C++ program to demonstrate lambda expression in C++; lambda expression allow us to write inline function  
 * 	@ref	http://www.geeksforgeeks.org/lambda-expression-in-c/ 
 * 	@details 
 * 
 * https://stackoverflow.com/questions/11604190/meaning-after-variable-type  
 * "&" meaning after variable type, means you're passing the variable by reference, 
 * The & means function accepts address (or reference) to a variable, instead of value of the variable.  
 * 
 * COMPILATION TIP : -Wall warning all, -g debugger, gdb can step through 
 * g++ -Wall -g methodconst.cpp -o methodconst
 * */
#include <iostream>  
#include <algorithm> // std::for_each, std::find_if, std::sort, std::count_if, std::unique  
#include <vector>
#include <numeric> // std::accumulate


/** @fn printVector
 * @brief Function to print vector 
 * */
void printVector(std::vector<int> v) {
	// lambda expression to print vector 
	std::for_each(v.begin(), v.end(), 
		[](int i) { std::cout << i << " "; } );
	std::cout << std::endl;	
}

int main() 
{
	std::vector<int> v { 4,1,3,5,2,3,1,7};
	
	printVector(v);
	
	// below snippet find first number greater than 4
	// find_if searches for an element for which 
	// function(third argument) returns true 
	std::vector<int>::iterator p = std::find_if(v.begin(), v.end(), 
		[](int i) { return i > 4; } ); 
		
	std::cout << "First number greater than 4 is : " << *p << std::endl; 

	// function to sort vector, lambda expression is for sorting in 
	// non-decreasing order Compiler can make out return type as 
	// bool, but shown here just for explanation
	std::sort(v.begin(), v.end(), 
		[](const int& a, const int& b) -> bool { return (a > b); });
		
	printVector(v);  
	
	// function to count numbers greater than or equal to 5
	int count_5 = std::count_if(v.begin(), v.end(), 
		[](int a) { return (a >= 5); });  

	std::cout << "The number of elements greater than or equal to 5 is : " 
		<< count_5 << std::endl;
		
	// function for removing duplicate element (after sorting all
	// duplicate comes together)
	p = std::unique(v.begin(), v.end(), 
			[](int a, int b) { return a == b; }); 
	
	
	// resizing vector to make size equal to total different number
	v.resize(distance(v.begin(), p) );
	printVector(v);  
	
	// accumulate function accumulate the container on the basis of 
	// function provided as third argument
	int arr[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
	int f = std::accumulate	(arr, arr+10, 1, 
		[](int i, int j) { return i * j ; } );
		
	std::cout << "Factorial of 10 is : " << f << std::endl;	

	// We can also access function by storing this into variable
	auto square = [](int i) { return i * i ; };
	
	std::cout << "Square of 5 is : " << square(5) << std::endl;

	/*
	 * Capturing ways  
	 * http://www.geeksforgeeks.org/lambda-expression-in-c/
	 */ 

	std::vector<int> v1 = {3, 1, 7, 9}; 
	std::vector<int> v2 = {10, 2, 7, 16, 9}; 
	
	// access v1 and v2 by reference  
	auto pushinto = [&] (int m) { 
									v1.push_back(m); 
									v2.push_back(m); }; 
	
	// it pushes 20 in both v1 and v2 
	pushinto(20);  
	
	// access v1 by copy 
	[v1]() {
		for (auto p = v1.begin(); p != v1.end(); p++) {
			std::cout << *p << " ";
		}
	};
	
	int N = 5;
	
	// below snippet find 1st number greater than N
	// [N] denotes, can access only N by value 
	std::vector<int>::iterator p1 = std::find_if(v1.begin(), v1.end(), 
		[N](int i) { return i>N; } );
		
	std::cout << "First number greater than 5 is : " << *p1 << std::endl; 
	
	// function to count numbers greater than or equal to N 
	// [=] denotes, can access all variable  
	int count_N = std::count_if(v1.begin(), v1.end(), 
		[=](int a) { return (a >= N); });
		
	std::cout << "The number of elements greater than or equal to 5 is : " 	
				<< count_N << std::endl; 
		

}
	
