// testing out arrays and multidimensional arrays in C++
/**
 * @ref : cf. 3.5. Arrays of Lippman, Lajoie, Moo, C++ Primer 5th ed.
 * */
#include <iostream>     // std::cout, std::fixed
#include <iomanip>      // std::setprecision
#include <cmath>        // expf, tanhf

#include <string>

float add_arr1d( const int N_in ) {
	// I can't do the following
//	constexpr int N_x {N_in };
//	constexpr int N_x = N_in ;
	constexpr int N_x { 5 };
	
	float arr1d[N_x];
	float sum { 0.f };
	for (auto i=0; i<N_x; ++i) {
		arr1d[i] = ((float) (i+1.f));
	}
	for (auto i=0; i<N_x; ++i) {
		sum += arr1d[i]; 
	}
	return sum;
}	

int main() {
	float DELTAt { 0.01f };
	float DELTAx { 0.0025f};
	float testval { DELTAt * (-1.f/2.f)*1.f/ (DELTAx) };

	std::cout << std::fixed;

	std::cout << " DELTAt : " ;
	std::cout << std::setprecision(9) <<  DELTAt << std::endl;

	std::cout << " DELTAx : " ;
	std::cout << std::setprecision(9) <<  DELTAx << std::endl;


	std::cout << " Test value : " ;
	std::cout << std::setprecision(9) <<  testval << std::endl;

	// 3.5.1. Defining and Initializing Built-in Arrays
	
	constexpr int size_int { 42 };
	constexpr unsigned size_uns { 69 };
	
	float arr[10] ;  // array of ten ints
	float *parr[size_int]; // array of 42 pointers to int
	
	float ux_constexprtest[ size_uns ];
	
	constexpr int N_x { 100 };
	float ux[N_x];
	for (auto i = 0; i < N_x; ++i) {
		ux[i] = tanhf( i ) ;
	}

	std::cout << " Results of tanh : " << std::endl;
	for (auto i = 0 ; i < 10; ++i) {
		std::cout << " " << ux[i] ;
	}

	float sumresult1 { add_arr1d(1) } ;
	float sumresult2 { add_arr1d(2) };
	float sumresult3 { add_arr1d(3) };
	float sumresult4 { add_arr1d(4) };

	std::cout << " Sum result 1 : " << sumresult1 << std::endl ;
	std::cout << " Sum result 2 : " << sumresult2 << std::endl;
	std::cout << " Sum result 3 : " << sumresult3 << std::endl;
	std::cout << " Sum result 4 : " << sumresult4 << std::endl;

	float *ptrs[10];            // ptrs is an array of ten pointers to int 
	float (*Parray)[10] = &arr; // Parray points to an array of ten ints
	float (&arrRef)[10] = arr;  // arrRef refers to an array of 10 ints
	float *(&arry)[10] = ptrs;  // arry is a reference to an array of 10 pointers
	
	/*
	 * 3.5.1. Defining and Initializing Built-in Arrays of Lippman, Lajoie, Moo (2012)
	 */
	unsigned cnt = 42; 				// not a constant expression
	constexpr unsigned sz = 42; 	// constant expression
									// constexpr see Sec. 2.4.4 (p. 66)
	int arr2[10]; 					// array of 10 ints
	int *parr2[sz]; 					// array of 42 pointers to int
	std::string bad[cnt];
	
	/*
	 * Explicitly Initializing Array Elements
	 */ 
	const unsigned sz2 = 3;
	int ia1[sz] = {0,1,2}; 			// array of 3 ints with values 0,1,2 
	int a2[] = {0,1,2}; 			// an array of dimension 3
	int a3[5] = {0,1,2}; 			// equivalent to a3[] = {0,1,2,0,0}
	std::string a4[3] = { "hi","bye"};	// same as a4[] = {"hi","bye",""}
	// int a5[2] = {0,1,2};			// error too many initializations

	/*
	 * Character Arrays are special, because we can initialize them with string literal, 
	 * i.e. string, and strings end with null character 
	 */
	char a1[] = {'C','+','+'};		// list initialization, no null
	char a2b[] = {'C','+','+','\0'}; // list initialization, explicit null
	char a3b[] = "C++";				// null terminator added automatically
//	const char a4b[6] = "Daniel"; 	// error: no space for the null!

	/*
	 * No copy or assignment; can't initialize array as copy of another array, 
	 * nor legal to assign 1 array to another 
	 */ 
	int a[] = {0,1,2}; 		// array of 3 ints
//	int a2c[] = a; 			// error: cannot initialize 1 array with another
	// a2 = a;
	
	/* Understanding Complicated Array Declarations */
	int *ptrsb[10];				// ptrs is an array of 10 pointers to int
	
	/* By default, type modifiers bind right to left.  
	 * Reading from inside out makes it much easier to understand type of Parray. 
	 */
	int (*Parray2)[10] = &arr2; 	// Parray points to an array of 10 ints  
	int (&arrRef2)[10] = arr2; 		// arrRef refers to an array of 10 ints

	/*
	 * 3.5.2. Accessing the Elements of an Array 
	 */ 

	 // count number of grades by clusters of 10
	unsigned scores[11] = {}; // 11 buckets, all values initialized to 0
	unsigned grade;
	while (std::cin >> grade) {
		if (grade <= 100) 
			++scores[grade/10]; 	// increment the counter for the current cluster
	}
	
	for (auto i : scores) 			// for each counter in scores
		std::cout << i << " "; 		// print the value of that counter
	std::cout << std::endl;

	/*
	 * 3.5.3. Pointers and Arrays
	 */ 
	std::string nums[] = {"one", "two", "three"};	// array of strings
	std::string *p = &nums[0]; 						// p points to the first element in nums

	/* 
	 * However, arrays have a special property - 
	 * in most places, when we use an array, 
	 * compiler automatically substitutes a pointer to the 1st element 
	 */
	std::string *p2 = nums; 		// equivalent to p2 = &nums[0]

	/* initialization */
	int ia[] = {0,1,2,3,4,5,6,7,8,9}; 	// ia is an array of 10 ints
	auto ia2(ia);						// ia2 is an int* that points to the 1st element in ia

	auto ia2b(&ia[0]);  				// now it's clear that ia2 has type int*
	
	// ia3 is an array of 10 ints
	decltype(ia) ia3 = {0,1,2,3,4,5,6,7,8,9};

	/*
	 * Pointers are iterators 
	 * pointers that address elements in an array have additional operations; in particular, 
	 * pointers to array elements support same operations as iterators on vectors or strings, 
	 * e.g. increment operator 
	 */

	int arr3[] = {0,1,2,3,4,5,6,7,8,9};
	int *p3 = arr3; 		// p points to the 1st element in arr
	++p3; 				// p points to arr[1]
	std::cout << " p3 : " << p3 << " p3[0] : " << p3[0] << std::endl;
	
	int *e = &arr3[10]; // pointer just past the last elementi n arr

	for (int *b = arr3; b != e; ++b) {
		std::cout << *b << std::endl;
	}
	auto b = arr3;
	b=b+4;
	std::cout << " b : " << b << " b[0] : " << b[0] << " *b : " << &b << std::endl;

	/*
	 * To make it easier and safer to use pointers, the new library includes 2 functions, 
	 * named begin and end.  
	 */
	int ia4[] = {0,1,2,3,4,5,6,7,8,9}; 	// ia is an array of 10 ints
	int *beg = std::begin(ia);			// pointer to the 1st element in ia
	int *last = std::end(ia); 			// pointer 1 past the last element in ia 
	std::cout << " beg : " << beg << " *beg : " << *beg << 
				" last : " << last << " *last : " << *last << 
				" (last -1) : " << (last - 1) << " *(last-1) : " << *(last-1) << std::endl;

	/*
	 * Using begin and end, it's easy to write loop to process the elements in an array. 
	 * For example, assuming arr is an array that holds int values, we might find the 
	 * first negative value in arr as follows:
	 */
	// pbeg points to the 1st and pend points just past the last element in arr
	int *pbeg = std::begin(arr3), *pend = std::end(arr3);
	// find the first negative element, stopping if we've seen all the elements 
	while (pbeg != pend && *pbeg >= 0) { 
		++pbeg;
	}
	std::cout << "\n pbeg : " << pbeg << " *pbeg : " << *pbeg << std::endl ;
	pbeg = pbeg -10;
	std::cout << " After pointer arithmetic of -10 - pbeg : " << pbeg << " *pbeg : " << *pbeg << std::endl ;
	
	while (pbeg != pend && *pbeg >= 0) { 
		std::cout << " pbeg : " << pbeg << " *pbeg : " << *pbeg << std::endl ;
		
		++pbeg;
	}

	/* 
	 * Pointer arithmetic 
	 * points to Table 3.6 (p. 107) and Table 3.7 (p. 111) 
	 */ 
	/* 
	 * When we add (or subtract) and integral value to (or from) a pointer, 
	 * the result is a new pointer 
	 */ 
	constexpr size_t sz3 = 5;
	int arr4[sz3] = {1,2,3,4,5}; 
	int *ip  = arr4; 		// equivalent to int *ip = &arr[0]
	int *ip2 = ip + 4;		// ip2 points to arr[4], the last element in arr 

	std::cout << " \n ip2 : " << ip2 << " ip2[0] : " << ip2[0] << " ip2[1] : " << ip2[1] << std::endl;

	/*
	 * as with iterators, subtracting 2 pointers gives us the distance between those pointers.  
	 * The pointers must point to elements in the same array:  
	 * The result of subtracting 2 pointers is a library type named ptrdiff_t 
	 * ptrdiff_t
	 * Like size_t, ptr_diff_t type is a machine-specific type and is defined in the cstddef header
	 * Because subtraction might yield a negative distance, ptrdiff_t is a signed integral type.  
	 */
	auto n = std::end(arr4) - std::begin(arr4); // n is 5, the number of elements in arr  
	std::cout << " \n n : " << n << std::endl;

	/*
	 * We can use the relational operators (e.g. < , > , <=, >= , etc.) 
	 * to compare pointers that point to elements of an array, or 1 past the last element in that array.  
	 * For example, we can traverse the elements in arr as follows: 
	 * We cannot use relational operators on pointers to 2 unrelated objects.  
	 */ 
	int *b2 = arr4, *e2 = arr4 + sz3; 
	while (b2<e2) {
		std::cout << " b2 : " << b2 << " *b2 : " << *b2 << std::endl;
		++b2;
	}

	/*
	 * Interaction between Dereference and Pointer Arithmetic 
	 * 
	 */
	int ia5[] = {0,2,4,6,8}; 		// array with 5 elements of type int 
	int last2 = *(ia5 + 4); 			// ok: initializes last to 8, the value of ia5[4] 

	last2 = *ia5 + 4; 				// ok: last = 4, equivalent to ia[0] +4

	/*
	 * Subscripts and Pointers
	 */
	int i = ia5[2] ; 				// ia5 is converted to a pointer to the 1st element in ia5
									// ia5[2] fetches element to which (ia5+2) points
	int *p4 = ia5;					// p4 points to 1st element in ia5
	i = *(p4 + 2);					// equivalent to i=ia5[2]

	int *p5 = &ia5[2];				// p5 points to the element indexed by 2
	int j =p5[1]; 					// p5[1] is equivalent to *(p5+1),
									// p5[1] is the same element as ia5[3]
	int k = p5[-2];					// p5[-2] is the same element as ia5[0]


}
