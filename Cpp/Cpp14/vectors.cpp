/* vectors.cpp
 * Ernest Yeung
 * ernestyalumni@gmail.com
 * Demonstrates vectors in C++11/C++14
 * cf. Lippman, Lajoie, Moo, C++ Primer 5th ed., 3.3. Library vector Type
 * */

#include <iostream>

#include <vector>

using std::vector; // std::vector -> vector

int main() {
	// cf. 3.3.1. Defining and Initializing vectors
	// vector holds objects of type T. Default initialization; vector is empty
	vector<float> flvec; 
	vector<unsigned int> univec;
	vector<vector<float>> fl2vec ;

	// vector<int> ivec(10,-1); // 10 int elements, each initialized to -1
	
	vector<float> flvecfin(10,0.f) ;
	vector<float> flvecfin_empty(10); // 10 elements, each initialized to 0
	vector<unsigned int> univecfin(10,0);
	vector<vector<float>> fl2vecfin( 10);

	// cf. 3.3.2. Adding Elements to a vector
	
	vector<int> v2; // empty vector
	for (int i = 0; i != 100; ++i) 
		v2.push_back(i);  // append sequential integers to v2
	// at end of loop v2 has 100 elements, values 0 ... 99
	
	for (int i =0; i != 10; ++i)
		std::cout << "This is v2, index: " << i << " entry: " << v2[i] << std::endl ;
		
	// cf. 3.3.3. Other vector Operations
	// cf. Table 3.5. vector Operations

	v2.empty(); // Returns true if v2 is empty; otherwise returns false	
	std::cout<< "Is v2 empty?  Do v2.empty() : " << v2.empty() << std::endl;

	v2.size(); // Returns the number of elements in v2
	std::cout << "Size of v2?  Do v2.size() : " << v2.size() << std::endl;
	
	
	std::cout << "Code snippet after Table 3.5. vector Operations " << std::endl;
	
	vector<int> v{1,2,3,4,5,6,7,8,9};
	for (auto &i : v) // for each element in v (note: i is a reference)
		i *= i;       // square the element value
	for (auto i : v)  // for each element in v
		std::cout << i << " ";
	std::cout << std::endl; 
	
	// EY : my try:
	std::cout << "Me playing with the referencing, deferencing in the range in the for loop " << std::endl;
	vector<int> vb{2,4,6,8,10,12};
	for (auto &i : vb) 
		std::cout << i << " ";
	std::cout << std::endl;
	// Indeed, C++ has implied deferencing that "makes sense" 
	for (auto i : v)
		std::cout << i << " ";
	std::cout << std::endl;
	
	for (auto i : vb)
		i += i ;
	for (auto i : vb)
		std::cout << i << " "; 
	std::cout << std::endl;  // 2 4 6 8 10 12 no change

	for (auto i : vb)
		i = 4 ;
	for (auto i : vb)
		std::cout << i << " "; 
	std::cout << std::endl;  // 2 4 6 8 10 12 no change

	// C++ is different from C in that the deferencing and referencing "makes sense"
	for (auto &i : vb)
		i += 4;
	for (auto i : vb)
		std::cout << i << " "; // 6 8 10 12 14 16 change!!!
	std::cout << std::endl;	
		
	// Computing a vector Index
	// EY : histogram
	
	vector<unsigned> scores(11, 0); // 11 buckets, all initially 0 
	unsigned grade;
	while (std::cin >> grade) { // read the grades
		if (grade <= 100)    // handle only valid grades
			++scores[grade/10];  // increment the counter for the current cluster
	}
	
	std::cout << " Histogram of scores inputted : " << std::endl;
	for (auto bin = 0 ; bin < 11 ; ++bin) { 
		std::cout << " For bin : " << bin << " score : " << scores[bin] << std::endl;
	}
	std::cout << std::endl;
	
	// 3.4. Introducing Iterators
	
	// Table 3.6. Standard Container Iterator Operations
	std::cout << "Using iterators " << std::endl;
	for (auto iter = v.begin(); iter != v.end(); ++iter) {
		// Returns a reference to the element denoted by the iterator iter.
		std::cout << *iter << " ";  // I found that it didn't work with iter
	
		// next line doesn't work, in that mem isn't there for int
//		std::cout << iter->mem << " ";  // Dereferences iter and fetches the member named mem from the underlying element.  Equivalent to (*iter).mem.
	}
	std::cout << std::endl;

	// Iterator Types
	vector<int>::iterator iiter; // iiter can read and write vector<int> elements
	vector<int>::const_iterator iiter3; // iiter3 can read but not write elements
	
	// The begin and end Operations
	
	vector<int> v3;
	const vector<int> cv;
	auto it1 = v3.begin() ; // it1 has type vector<int>::iterator
	auto it2 = cv.begin() ; // it2 has type vector<int>::const_iterator 
	
	auto it3 = v3.cbegin() ; // it3 has type vector<int>::const_iterator

	// Combining Dereference and Member Access
	
//	(*iiter).empty() ;

	for (auto iter = scores.cbegin(); 
	// I obtain an error when using empty on an iterator
//			iter != scores.cend() && !(iter->empty()) ; ++iter) {
		iter != scores.cend()  ; ++iter) {
		std::cout << *iter << std::endl;
	}
	
	// 3.4.2. Iterator Arithmetic 
	
	// compute an iterator to the element closest to the midpoint of vi
	auto mid = v2.begin() + v2.size() / 2;
	
	// EY : I'm trying stuff out myself
	auto firstten = v2.begin() + 10; 
	
	auto itertrial = v2.begin(); 
	
	while (itertrial != firstten ) {
		std::cout << " This is in v2 : " << *itertrial << std::endl;
		++itertrial;
	}
	
	// Using Iterator Arithmetic
		/* 
		 * binary search is a classica algorithm!!!
		 * Now let's do binary search using iterators!
		 * */
	
	

}
