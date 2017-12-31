/**
 * @file   : fstream_eg.cpp
 * @brief  : fstream example 
 * @details : fstream example. fstream is a "class template instance"    
 * | is bitwise OR, takes 2 bit patterns of equal length and performs logical inclusive OR on each pair of bits   
 * @author : Ernest Yeung <ernestyalumni@gmail.com>
 * @date   : 20171230    
 * @ref    : http://en.cppreference.com/w/cpp/io/basic_fstream
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
 * g++ main.cpp ./structs/structs.cpp -o main
 * 
 * */

#include <iostream>   
#include <fstream>  
#include <string>  

int main() {
	std::string filename = "test.bin";  
	std::fstream s(filename, s.binary | s.trunc | s.in | s.out); // | is bitwise OR 
	if (!s.is_open()) { 
		std::cout << "failed to open " << filename << '\n'; 
	} else {
		// write 
		double d = 3.14; 
		s.write(reinterpret_cast<char*>(&d), sizeof d); // binary output  
		s << 123 << "abc"; 								// text output  
		
		// for fstream, this moves the file position pointer (both put and get)  
		s.seekp(0);  
	
		// read  
		s.read(reinterpret_cast<char*>(&d), sizeof d); // binary input
		int n; 
		std::string str; 
		if (s>> n >> str) { 
			std::cout << "read back from file: " << d << ' ' << n << ' ' << str << '\n'; 
		}
	} // end of else	

	
}
