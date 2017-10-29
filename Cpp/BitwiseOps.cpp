/**
 * @file   : BitwiseOps.cpp
 * @author : Ernest Yeung	ernestyalumni@gmail.com
 * @date   : 20170524
 * @brief	: Demonstrates bitwise operations
 * @ref    : cf.  http://www.geeksforgeeks.org/interesting-facts-bitwise-operators-c/
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
 * P.S. I'm using an EVGA GeForce GTX 980 Ti which has at most compute capability 5.2; 
 * A hardware donation or financial contribution would help with obtaining a 1080 Ti with compute capability 6.2
 * so that I can do -arch='sm_62' and use many new CUDA Unified Memory features
 * */
 /**
  * COMPILATION TIP(s)
  * (without make file, stand-alone)
  * g++ -std=c++14 BitwiseOps.cpp -o BitwiseOps.exe
  * 
  * */
#include <iostream>

#include <vector>

#include <stdio.h>

int findOdd(int arr[], int n) {
	int res = 0;
	int i;
	for (i =0; i<n;i++) {
		res ^= arr[i]; }
	return res;
};

int findOdd_pedagogical(int arr[], int n) {
	int res = 0;
	int i;
	for (i =0; i<n;i++) {
		printf(" %d : ", res^arr[i]);
		res ^= arr[i]; 
		printf(" %d , ", res);
		}
		
	return res;
};

// Find the missing number
int getMissingNumber(int arr[], int n) {
	int res = 0;
	for (int i=0; i<n; i++ ) { 
		auto ele = arr[i];
		printf(" %d : ", res^ele);
		res ^= ele;
		printf(" %d , ",res);
	}
	std::cout << std::endl;

	int res2 = 0;
	for (int i=1;i<=(n+1); i++) {
		printf(" %d : ", res2^i);
		res2^=i;
		printf(" %d : ", res2);
	}
	return res^res2;
}	


int main(int argc, char* argv[]) {
	
	// from "Interesting Facts about Bitwise Operators in C" of 
	// GeeksforGeeks, 
	// cf. http://www.geeksforgeeks.org/interesting-facts-bitwise-operators-c/
	
	unsigned char a = 5, b = 9; // a = 4(00000101), b = 8(00001001)
	std::cout << "a = " << static_cast<unsigned>(a) << " b = " << 
		static_cast<unsigned>(b) << std::endl;
	auto result_AND = a&b;
	std::cout << "a&b = " << static_cast<unsigned>(result_AND) << std::endl; // The result is 1
	std::cout << "a|b = " << static_cast<unsigned>(a|b) << std::endl; // The result is 13

		// ^ (bitwise XOR) - result of XOR is 1 if 2 bits are different
	std::cout << "a^b = " << static_cast<unsigned>(a^b) << std::endl; // The result is 12

		// ~ (bitwise NOT) Takes 1 number and inverts all bits of it
	std::cout << "~a = " << static_cast<unsigned>(~a) << std::endl; // The result is 4294967290
	std::cout << "~a = " << static_cast<int>(~a) << std::endl; // The result is -6


	std::cout << "a<<1 = " << static_cast<unsigned>(a<<1) << std::endl; // The result is 10
	std::cout << "a>>1 = " << static_cast<unsigned>(a>>1) << std::endl; // The result is 2

		// The bitwise XOR operator is the most useful operator, technically; It's used in many problems
		/* e.g. Given a set of numbers where all elements occur even number of times, 
		 * 		find the odd occurring number."
		 * */
	int arr[] = {12, 12, 14, 90, 14, 14, 14}; 
	int n = sizeof(arr)/sizeof(arr[0]); 
	std::cout << "The odd occurring element is " << 
		static_cast<unsigned>(findOdd(arr,n)) << std::endl; 
	
	auto test_res = findOdd_pedagogical(arr,n);

		// 5) The & operator can be used to quickly check if a number is odd or even
	int x = 19;
	(x & 1) ? std::cout << "Odd" : std::cout << "Even " ;
	std::cout << std::endl; 
	x = 22;
	(x & 1) ? std::cout << "Odd" : std::cout << "Even" ; 
	std::cout << std::endl;
	x = 35;
	(x & 1) ? std::cout << "Odd" : std::cout << "Even" ;
	std::cout << std::endl;

	// cf. http://www.geeksforgeeks.org/swap-two-numbers-without-using-temporary-variable/
	// swap 2 numbers without using a third variable
	x = 10;
	int y = 5;
	
	x = x ^ y; 
	std::cout << x << std::endl;
	y = x ^ y;
	std::cout << y << std::endl;
	x = x ^ y;
	std::cout << x << std::endl;
	
	// Find the Missing Number
	// cf. http://www.geeksforgeeks.org/find-the-missing-number/
	
	int arra[] = {1,2,4,5,6};
	int miss = getMissingNumber(arra,5);
	std::cout << " \n Missing : " << miss << std::endl;

	int arrb[] = {1,8,2,4,3,6,7};
	miss = getMissingNumber(arrb,7);
	std::cout << " \n Missing : " << miss << std::endl;

	constexpr const int WARPSIZE = 32;

	for (int i=0; i<99; i++) { 
		printf(" %d : %d ", i,i&WARPSIZE); }
	
	std::cout << std::endl;
	std::cout << std::endl;
	
	for (int i=0; i<133; i++) { 
		printf("%d ", i & (WARPSIZE-1)); }
	

	std::vector<bool> boolvec { true, false, true, true };
	std::vector<float> fvec;

	std::cout << std::endl << std::endl;  	
	
	for (auto ele : boolvec) {
		std::cout << ele << " " ;
		fvec.push_back( ((float) ele ) ); }
	std::cout << std::endl << std::endl;  	
	for (auto ele : fvec) {
		std::cout << ele << " " ; }	
	
	
	getchar();
	
}
