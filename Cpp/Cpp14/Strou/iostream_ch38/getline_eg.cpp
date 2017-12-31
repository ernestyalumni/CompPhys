/**
 * @file   : getline_eg.cpp
 * @brief  : getline example 
 * @details : getline example.  getline  
 * @author : Ernest Yeung <ernestyalumni@gmail.com>
 * @date   : 20171231    
 * @ref    : http://en.cppreference.com/w/cpp/io/basic_istream/getline
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
#include <iostream> 
#include <sstream>  // std::istringstream
#include <vector>  
#include <array>  

int main()
{
	std::istringstream input("abc|def|gh");  
	std::vector<std::array<char, 4>> v;   
	
	// note: the following loop terminates when std::ios_base::operator bool()  
	// on the stream returned from getline() returns false  
	for (std::array<char, 4> a; input.getline(&a[0], 4, '|'); ) {
		v.push_back(a); 
	}
	
	for (auto& a: v) {
		std::cout << &a[0] << '\n'; 
	}
}
	
