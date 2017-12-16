/**
 * 	@file 	genericlambda14.cpp
 * 	@brief 	Return type deduction    
 * 	@ref	http://www.drdobbs.com/cpp/the-c14-standard-what-you-need-to-know/240169034
 * 	@details compiler deduces what type, 
 * Use auto function return type to return complex type, such as iterator, 2. refactor code     
 * 
 * COMPILATION TIP : -Wall warning all, -g debugger, gdb can step through 
 * g++ -Wall -g copyctordemo.cpp -o copyctordemo
 * */
#include <iostream>
#include <vector>
#include <string>
#include <numeric>  

int main() 
{
	std::vector<int> ivec = { 1, 2, 3, 4};
	std::vector<std::string> svec = { 	"red", 
										"green",
										"blue" };
	auto adder = [](auto op1, auto op2) { return op1 + op2; };
	std::cout << "int result : " 
				<< std::accumulate(ivec.begin(),
									ivec.end(),
									0, 
									adder )
				<< "\n"; 
	std::cout << "string result : "  
				<< std::accumulate(svec.begin(), 
									svec.end(),
									std::string(""),
									adder )
				<< "\n";

	/**
	 * @details employing generic parameters, i.e. defining lambda parameters with auto type declaration, 
	 * useful for instantiating anonymous inline lambdas
	 */
	std::cout << "string result : " 
				<< std::accumulate(svec.begin(), 
									svec.end(),
									std::string(""),
									[](auto op1, auto op2) {return op1+op2; } ) 
				<< "\n";

	return 0;								
}
