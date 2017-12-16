/**
 * 	@file 	returndeduct14.cpp
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
#include <algorithm> // std::find_if

auto f(int i) {
	if (i<0) { 
//		return -1; // error: inconsistent deduction for auto return type: ‘int’ and then ‘double’
//		return -1.f; // error: inconsistent deduction for auto return type: ‘float’ and then ‘double’
		return -1.0; 
	}
	else {
		return 2.0;
	}
}

/**
 * @details refactor with auto function return type  
 * */
struct record {
	std::string name;
	int id;
}; 

auto find_id(	const std::vector<record> &people, 
				const std::string &name) {
	auto match_name = [&name](const record& r) -> bool {
		return r.name == name;
	};
	auto ii = std::find_if(people.begin(), people.end(), match_name );
	if (ii == people.end()) {
		return -1; 
	}
	else {
		return ii->id;
	}
}

int main() 
{
	std::cout << " f(-3) : " << f(-3) << " f(2) : " << f(2) << std::endl;

	std::vector<record> roster = { 	{"mark",1},
									{"bill",2},
									{"ted",3}};
	std::cout << find_id(roster,"bill") << "\n";
	std::cout << find_id(roster,"ron") << "\n";
	
}
