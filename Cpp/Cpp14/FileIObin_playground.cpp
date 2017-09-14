/**
 * @file   : FileIObin_playground.cpp
 * @brief  : Playground for Examples to demonstrate binary file IO in C/C++, especially C++11/14
 * 			: since there's many new C++11/14 features  
 * @author : Ernest Yeung <ernestyalumni@gmail.com>
 * @date   : 20170913
 * @ref    : https://stackoverflow.com/questions/11563963/writing-a-binary-file-in-c-very-fast
 * 				Writing a binary file in C++ very fast
 * 			: http://en.cppreference.com/w/cpp/types/integer
 * 				Fixed width integer types (since C++11)
 * 			: http://en.cppreference.com/w/cpp/types/size_t
 * 				std::size_t
 * 
 * If you find this code useful, feel free to donate directly and easily at this direct PayPal link: 
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
 * g++ FileIObin.cpp -o FileIObin.exe
 * 
 * */
 
// I confirmed that std::size_t belongs in cstdint, even though http://en.cppreference.com/w/cpp/types/size_t doesn't say so
#include <cstdint> 		// std::uint64_t, std::int64_T, std::sizeof
#include <iostream>		// std::cout

int main(int argc, char **argv) {
	std::cout << " sizeof(std::uint64_t) : " << sizeof(std::uint64_t) << std::endl;		// 8
	std::cout << " sizeof(std::int64_t)  : " << sizeof(std::int64_t) << std::endl;		// 8

	// cf. http://en.cppreference.com/w/cpp/types/size_t
	const std::size_t N = 10; 
	int * a = new int[N];
	
	for (std::size_t n=0; n< N; ++n) { a[n] = n; }
	for (std::size_t n=N; n-- > 0;) // Reverse cycles are tricky for unsigned types.  
		std::cout << a[n] << " ";
	
	/*
	 * https://stackoverflow.com/questions/28338775/what-is-iosiniosout
	 * 

    ios::in allows input (read operations) from a stream.
    ios::out allows output (write operations) to a stream.
    | (bitwise OR operator) is used to combine the two ios flags,
    meaning that passing ios::in | ios::out to the constructor
    of std::fstream enables both input and output for the stream.

Important things to note:

    std::ifstream automatically has the ios::in flag set.
    std::ofstream automatically has the ios::out flag set.
    std::fstream has neither ios::in or ios::out automatically
    set. That's why they're explicitly set in your example code.


	 * */
	 
	 
	
	
	// free any resources
	delete[] a;
	
	return 0;
}
