/**
 * 	@file 	optional.cpp
 * 	@brief 	class template std::optional manages an optional contained value, i.e. value that may or may not be present
 * 	@ref	http://en.cppreference.com/w/cpp/utility/optional 
 * 	@details Common use case for optional is return value of function that may fail.  
 * optional handles expensive-to-construct objects well and is more readable, as intent expressed explicitly.   
 * 
 * COMPILATION TIP : -Wall warning all, -g debugger, gdb can step through 
 * g++ -Wall -g methodconst.cpp -o methodconst
 * */
#include <string> 
#include <iostream>  
#include <optional>

// optional can be used as the return type of a factory that may fail
std::optional<std::string> create(bool b) {
	if (b) {
		return "Godzilla"; }
	return {};
}  

// std::nullopt can be used to create any (empty) std::optional 
auto create2(bool b) {
	return b ? std::optional<std::string>{ "Godzilla" } : std::nullopt; 
}

int main() 
{
	std::cout << "create(false) returned " 
				<< create(false).value_or("empty") << '\n'; 
				
	// optional-returning factory functions are usable as conditions of while and if 
	if (auto str = create2(true)) {
		std::cout << "create2(true) returned " << *str << '\n'; 
	}
}
