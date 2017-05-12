/**
 * @file   : OddEvenSort.cpp
 * @author : Ernest Yeung	ernestyalumni@gmail.com
 * @date   : 20170512
 * @ref    : cf. http://www.personal.kent.edu/~rmuhamma/Algorithms/MyAlgorithms/Sorting/quickSort.htm
 * http://stackoverflow.com/questions/22504837/how-to-implement-quick-sort-algorithm-in-c
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
  * g++ -std=c++14 OddEvenSort.cpp -o OddEvenSort.exe
  * 
  * */

#include <iostream>
#include <vector>

// To make interesting boilerplate
#include <random> // random_device, mt19937, uniform_int_distribution
#include <functional> // bind, needed for generate
#include <algorithm> // generate

template <typename T>
void OddEvenSort_even(std::vector<T> & A_even, const int N_even, const int MAX_ITERS) {
	volatile bool FLAG_FIXED_PT_01 { false }; 
	volatile bool FLAG_FIXED_PT_02 { false }; 

	const int M_even = N_even / 2; 

	int iter=0;
	while ((FLAG_FIXED_PT_02 == false) && (iter < MAX_ITERS) ) { 
		FLAG_FIXED_PT_01 = true;
		
		// even-pairing sort
		for (int j = 0; j < M_even; j++) {
			if (A_even[2*j] > A_even[2*j+1]) { 
				std::swap( A_even[2*j] , A_even[2*j+1] ); 
			
				// let us know that a swap had occured 
				FLAG_FIXED_PT_01 = false;
			}		
		}

		// odd-pairing sort
		for (int j = 0; j < M_even-1; j++) {
			if (A_even[2*j+1] > A_even[2*j+2]) { 
				std::swap( A_even[2*j+1] , A_even[2*j+2] ); 
				
			// let us know that a swap had occured 
			FLAG_FIXED_PT_01 = false;
			}

		}
		if (FLAG_FIXED_PT_01) {
			FLAG_FIXED_PT_02 = true; }

		iter+=1 ; 
		
	}
	return ; 
}

template <typename T>
void OddEvenSort_odd(std::vector<T> & A_odd, const int N_odd, const int MAX_ITERS) {
	volatile bool FLAG_FIXED_PT_01 { false }; 
	volatile bool FLAG_FIXED_PT_02 { false }; 

	const int M_odd = N_odd / 2; 

	int iter=0;
	while ((FLAG_FIXED_PT_02 == false) && (iter < MAX_ITERS) ) { 
		FLAG_FIXED_PT_01 = true;
		
		// even-pairing sort
		for (int j = 0; j < M_odd; j++) {
			if (A_odd[2*j] > A_odd[2*j+1]) { 
				std::swap( A_odd[2*j] , A_odd[2*j+1] ); \
						
				// let us know that a swap had occured 
				FLAG_FIXED_PT_01 = false;
			}		
		}

		// odd-pairing sort
		for (int j = 0; j < M_odd; j++) {
			if (A_odd[2*j+1] > A_odd[2*j+2]) { 
				std::swap( A_odd[2*j+1] , A_odd[2*j+2] ); 
						
			// let us know that a swap had occured 
			FLAG_FIXED_PT_01 = false;
			}

		}
		if (FLAG_FIXED_PT_01) {
			FLAG_FIXED_PT_02 = true; }

		iter+=1 ; 
		
	}
	return ; 
}


int main() 
{
	constexpr const int M { 20 };
	constexpr const int N_even = 2*M;
	constexpr const int N_odd = 2*M + 1;
	
	// boilerplate for random but interesting values to test on 
	constexpr const int min_val = 3;
	constexpr const int max_val = 42*4;
	std::vector<int> A_even( N_even) ;
	std::vector<int> A_odd(  N_odd ) ;
	
	std::cout << N_even % 2 << std::endl; 
	std::cout << N_odd % 2 << std::endl; 
	
	std::random_device rd; // Will be used to obtain a seed for the random number engine
	std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()

	std::uniform_int_distribution<int> randomInteger(min_val,max_val);
	auto randomIntegerGen = std::bind(randomInteger, gen);
	std::generate(A_even.begin(),A_even.end(), randomIntegerGen); 
	std::generate(A_odd.begin(),A_odd.end(), randomIntegerGen); 
	
	std::cout << "====== Original =======" << std::endl;
	for (auto e: A_even) { 
		std::cout << e << " " ; }
	std::cout << std::endl; 
	
	for (auto e: A_odd) { 
		std::cout << e << " " ; }
	std::cout << std::endl; 
	
	// playground 
	const int M_even = A_even.size()/2;
	const int M_odd  = A_odd.size() /2;
	std::cout << " M_even : " << M_even << std::endl;
	std::cout << " M_odd  : " << M_odd  << std::endl;
	
	for (int j = 0; j < M_even; j++) {
		std::cout << A_even[2*j] << A_even[2*j+1] << " "; } std::cout << std::endl; 
	for (int j = 0; j < M_odd; j++) {
		std::cout << A_odd[2*j] << A_odd[2*j+1] << " "; } std::cout << std::endl; 

	// Even-pairing sort
	for (int j = 0; j < M_even; j++) {
		if (A_even[2*j] > A_even[2*j+1]) { 
			std::swap( A_even[2*j] , A_even[2*j+1] ); }
	}
	
	for (int j = 0; j < M_odd; j++) {
		if (A_odd[2*j] > A_odd[2*j+1]) { 
			std::swap( A_odd[2*j] , A_odd[2*j+1] ); }
	}
		
	for (auto e: A_even) { 
		std::cout << e << " " ; }
	std::cout << std::endl; 
	
	for (auto e: A_odd) { 
		std::cout << e << " " ; }
	std::cout << std::endl; 

	// Odd-pairing sort	
	for (int j = 0; j < M_even-1; j++) {
		std::cout << A_even[2*j+1] << A_even[2*j+2] << " "; } std::cout << std::endl; 
	for (int j = 0; j < M_odd; j++) {
		std::cout << A_odd[2*j+1] << A_odd[2*j+2] << " "; } std::cout << std::endl; 

	for (int j = 0; j < M_even-1; j++) {
		if (A_even[2*j+1] > A_even[2*j+2]) { 
			std::swap( A_even[2*j+1] , A_even[2*j+2] ); }
	}
	
	for (int j = 0; j < M_odd; j++) {
		if (A_odd[2*j+1] > A_odd[2*j+2]) { 
			std::swap( A_odd[2*j+1] , A_odd[2*j+2] ); }
	}

	for (auto e: A_even) { 
		std::cout << e << " " ; }
	std::cout << std::endl; 
	
	for (auto e: A_odd) { 
		std::cout << e << " " ; }
	std::cout << std::endl; 

	OddEvenSort_even<int>(A_even, N_even, 1000);
	OddEvenSort_odd<int>(A_odd, N_odd, 1000);


	std::cout << "======== Sorted =======" << std::endl;
	for (auto e: A_even) { 
		std::cout << e << " " ; }
	std::cout << std::endl; 

	for (auto e: A_odd) { 
		std::cout << e << " " ; }
	std::cout << std::endl; 




}
	
	
	


