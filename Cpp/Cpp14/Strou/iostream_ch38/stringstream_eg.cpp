/**
 * @file   : stringstream_eg.cpp
 * @brief  : stringstream example; swapping ostringstream objects 
 * @details : stringstream example.  stringstream  
 * @author : Ernest Yeung <ernestyalumni@gmail.com>
 * @date   : 20171231    
 * @ref    : http://www.cplusplus.com/reference/sstream/stringstream/stringstream/
 * Ch. 38 Bjarne Stroustrup, The C++ Programming Language, 4th Ed. 2013 Addison-Wesley 
 * 
 * https://www.paypal.com/cgi-bin/webscr?cmd=_donations&business=ernestsaveschristmas%2bpaypal%40gmail%2ecom&lc=US&item_name=ernestyalumni&currency_code=USD&bn=PP%2dDonationsBF%3abtn_donateCC_LG%2egif%3aNonHosted 
 * 
 * which won't go through a 3rd. party such as indiegogo, kickstarter, patreon.  
 * Otherwise, I receive emails and messages on how all my (free) material on 
 * physics, math, and engineering have helped students with their studies, 
 * and I know what it's like to not have money as a student, but love physics 
 * (or math, sciences, etc.), so I am committed to keeping all my material 
 * open-source and free, whether or not 
 * sufficiently crowdfunded, under the open-source MIT license: 
 * 	feel free to copy, edit, paste, make your own versions, share, use as you wish.  
 *  Just don't be an asshole and not give credit where credit is due.  
 * Peace out, never give up! -EY
 * 
 * */
/* 
 * COMPILATION TIP
 * g++ getline_eg.cpp -o getline_eg
 * 
 * */
#include <string> 		// std::string
#include <iostream>		// std::cout
#include <sstream>		// std::stringstream  

int main() {
	
	std::stringstream ss;  
	
	ss << 100 << ' ' << 200; 
	
	int foo,bar; 
	ss >> foo >> bar;  
	
	std::cout << "foo: " << foo << '\n'; 
	std::cout << "bar: " << bar << '\n';  

	std::stringstream ss1;  

	ss1 << 10.0 << " " << 20.0; 
	
	double foo1,bar1; 
	ss1 >> foo1; 
	ss1 >> bar1; 
	
	std::cout << "foo1: " << foo1 << std::endl; 
	std::cout << "bar1: " << bar1 << std::endl;
	
	
	return 0; 
}
	
