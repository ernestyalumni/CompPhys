/* vectors_binarysearch.cpp
 * Ernest Yeung
 * ernestyalumni@gmail.com
 * Demonstrates binary search with vectors in C++11/C++14
 * cf. Lippman, Lajoie, Moo, C++ Primer 5th ed., 3.4. Introducing Iterators
 * Using Iterator Arithmetic
 * */

#include <iostream>
#include <vector>

// this is boilerplate to have something to work with, so I utilized random library
#include <random>

using std::vector; // std::vector -> vector

int main() 
{
	constexpr int VECTORSIZE { 100 } ; 
	
	// "boilerplate" so that we can make an interesting vector to binary search over
	// cf. http://en.cppreference.com/w/cpp/numeric/random/uniform_int_distribution
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> uni_int_dis(1,100) ; // default is int, inside <>
	
	for (int n=0; n< 10; ++n) 
		std::cout << uni_int_dis(gen) << ' ';
	std::cout << "\n";

	auto target_index = uni_int_dis(gen); // this is where we want to find, amongst entries in vector
	std::cout << " This is the target index : " << target_index << std::endl;

	// set up the vector with interesting entries
	vector<unsigned> input_vector(VECTORSIZE,0) ; 
	
	for (auto &iter : input_vector) { 
		iter = uni_int_dis(gen) ; 
	}
	
	auto firstten = input_vector.begin() + 10; 
	auto iter     = input_vector.begin(); 
	
	while (iter != firstten ) {
		std::cout << " This is in input_vector : " << *iter << std::endl;
		++iter;
	}
	
	input_vector[ target_index - 1] = 101 ; // set the value to be found; 101 choice is arbitrary
	// sanity check
	std::cout << " This is the value at the target_index of input_vector : " << 
		input_vector[ target_index -1 ] << std::endl ; 
	
	// END of boilerplate and vector setup
	
	// cf. Lippman, Lajoie, Moo, C++ Primer 5th Ed.
	// 3.4.2. Iterator Arithmetic, Using Iterator Arithmetic 
	// beginning of the "MEAT" - binary search (classic algorithm) with C++11/14 vectors

// original code snippet	
	auto beg = input_vector.begin(), end = input_vector.end() ; 
	auto mid = input_vector.begin() + (end - beg)/2; // original midpoint
	// while there are still elements to look at and we haven't yet found sought
	unsigned sought { 101 }; // sought = 101 is arbitrary
	// sanity check:
	std::cout << " Is sought the same as the target_index value? : " <<
		( input_vector[ target_index - 1] == sought ) << std::endl;

 	while (mid != end && *mid != sought) { 
		if (sought < *mid) // is the element we want in the first half?
			end = mid;  // if so, adjust the range to ignore the second half
		else // the element we want is in the second half
			beg = mid + 1; // start looking with the element just after mid
		mid = beg + (end - beg)/2; // new midpoint
	}
	
	std::cout << " This is the value of beg after all is said and done : " << *beg << std::endl;
	std::cout << " This is the value of mid after all is said and done : " << *mid << std::endl;
	std::cout << " This is the value of end after all is said and done : " << *end << std::endl;

// EY my code snippet
	beg = input_vector.begin(), end = input_vector.end() ;
	mid = input_vector.begin() + (end - beg)/2;

	int flag = 0;
 	while (mid != end && *mid != sought) { 
		flag = 0;
		for (auto iter = beg; iter != mid; ++iter) {
			if ( *iter == sought) 
				flag = 1;
		}
		if (flag == 1)
			end = mid;
		else
			beg = mid + 1;
		mid = beg + (end - beg)/2;
	}

	std::cout << " This is the value of beg after all is said and done : " << *beg << std::endl;
	std::cout << " This is the value of mid after all is said and done : " << *mid << std::endl;
	std::cout << " This is the value of end after all is said and done : " << *end << std::endl;

	std::cout << " If mid is 101, we've found it! (here is (*mid == sought) ): " 
		<< (*mid == sought) << std::endl;
	
}

