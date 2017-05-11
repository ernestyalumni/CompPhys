/**
 * @file   : quicksortEY.cpp
 * @author : Ernest Yeung	ernestyalumni@gmail.com
 * @date   : 20170511
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
  * g++ -std=c++14 quicksortEY.cpp -o quicksortEY.exe
  * 
  * */

#include <iostream>
#include <vector>

#include <random> // random_device, mt19937, uniform_int_distribution
#include <algorithm> // generate
#include <functional> // bind

void quickSort(std::mt19937, std::vector<int>&, int, int);

void randomized_quickSort(std::mt19937, std::vector<int>&, int, int);


int partition(std::vector<int>&, int, int );

int randomized_partition(std::mt19937, std::vector<int>&, int, int );


int main() 
{
	
	// testing out std::uniform_int_distribution
	std::random_device rd; // Will be used to obtain a seed for the random number engine
	std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
	std::uniform_int_distribution<int> randomInteger_test(2,7); // this will include the inputs as well for possible random values
	
	for (auto i=0; i<10;i++){ 
		std::cout << randomInteger_test(gen) << " "; } std::cout << std::endl; 

	// boilerplate for vector initialization
	int N = 20;
	int min_val = 3;
	int max_val = 42;
	std::vector<int> A(N);
	int p = 0;
	int q = A.size();

	std::uniform_int_distribution<int> randomInteger(min_val, max_val);
	auto randomIntegerGen = std::bind( randomInteger, gen) ; 
	std::generate(A.begin(), A.end(), randomIntegerGen); 

	
	std::cout << "====== Original =======" << std::endl;
	for (auto e: A) { 
		std::cout << e << " " ; }
	std::cout << std::endl; 

	// testing out partition
	partition(A, 2,6);
	for (auto e: A) { 
		std::cout << e << " " ; }
	std::cout << std::endl; 
	
	
	randomized_quickSort(gen, A, p,q);
	
	std::cout << "======== Sorted =======" << std::endl;
	for (auto e: A) { 
		std::cout << e << " "; }
	std::cout << std::endl; 
	
}


void quickSort(std::vector<int>& A, int p, int q) 
{
	int r; 
	if (p<q) {
		r = partition(A,p,q);
		quickSort(A,p,r);
		quickSort(A,r+1,q);
	}
}

void randomized_quickSort(std::mt19937 gen, std::vector<int>& A, int p, int q) {
	int r;
	if (p<q) { 
		r = randomized_partition(gen, A, p,q); 
		randomized_quickSort(gen, A, p ,r);
		randomized_quickSort(gen,A,r+1,q);
	}
}

	
int partition(std::vector<int>& A, int p, int q) {
	int x = A[p];
	int i = p;
	int j;

	for (int j=p+1; j<q;j++) { 
		if ( A[j] <= x ) { 
			i += 1;
			std::swap(A[i],A[j]); 
		}
	}
	
	std::swap(A[i],A[p]);
	return i;
}

int randomized_partition(std::mt19937 gen, std::vector<int>& A, int p, int q) {
	std::uniform_int_distribution<int> randomInteger(p,q-1);
	int i = randomInteger(gen);
	std::swap(A[i],A[p]);
	i = partition(A, p,q);
	return i;
}
	

