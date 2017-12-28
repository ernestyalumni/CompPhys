/**
 * @file   : structs.h
 * @brief  : structs in header file, in C++14, 
 * @details : struct
 * 
 * @author : Ernest Yeung <ernestyalumni@gmail.com>
 * @date   : 20171227    
 * @ref    : Ch. 8 Structures, Unions, and Enumerations; Bjarne Stroustrup, The C++ Programming Language, 4th Ed.  
 * Addison-Wesley 
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
#ifndef __STRUCTS_H__
#define __STRUCTS_H__ 

#include <iostream>
#include <vector> // std::vector
#include <string> 

#include <sstream> // ostringstream

/**
 * @struct Address_hrd
 * @brief Address in the header, completely in the header   
 * @details struct is aggregate of elements of arbitrary types 
 */
struct Address_hdr { 
	const char* name;		// "Jim Dandy"
	int number;				// 61 
	const char* street;		// "South St"
	const char* town;		// "New Providence"
	char state[2];			// 'N' 'J'
	const char* zip; 		// "07974"
};  

/**
 * @struct Address_sep
 * @brief struct Address declaration in header, implementation/content/body separate
 * @details struct is aggregate of elements of arbitrary types 
 * @ref https://stackoverflow.com/questions/15786902/c-structs-defined-in-header-and-implemented-in-cpp
 * BIG NOTE you can't have an incomplete type; all forward declaration of struct's members and member functions 
 * must be here
 */
// struct Address_sep; // error incomplete type

/** @fn f
 * @details variables of type Address (newly defined struct) can be declared like other variables, and 
 * individual members can be accessed using . (dot) operator
 * Uses Address_hdr struct  
 * */
void f(); 

/** @fn f
 * @details variables of type Address (newly defined struct) can be declared like other variables, and 
 * individual members can be accessed using . (dot) operator
 * Uses Address_sep struct  
 * */
void f_sep(); 

/** @fn print_addr
 *  @details Structures often accessed through pointers using -> (struct pointer dereference) operator 
	 * */ 
void print_addr(Address_hdr* p) ;


/** @fn print_addr2
 * @details a struct can be passed by reference and accessed using the . (struct member access) operator 
 * */
void print_addr2(const Address_hdr& r);	

/** 
 * Objects of structure types can be assigned, passed as function arguments, and 
 * returned as a result from a function.  Example: 
 * @details static Address_hdr current_global compiles, but at run-time, 
 * when applying function to assign values to it, program stops  
 * */
static Address_hdr current_global;



Address_hdr set_current(Address_hdr next) ;

struct Link { 
	Link* previous;
	Link* successor; 
};  


/* ========== 8.2.3 Structures and Classes ========== */
/* ===== struct is simply a class where the members are public by default.  
 * So struct can have member functions; in particular, a struct can have constructors */ 
/** @struct Point 
 * @details Point struct
 */
struct Point { 
	int x,y; 
};

/** @struct Points 
 * @details Points struct, using Point struct
 * */
struct Points {
	std::vector<Point> elem;	// must contain at least 1 Point_main

	// constructors
//	Points(Point p0) { elem.push_back(p0) ; }  // WORKS

	Points(Point p0) ;
	Points(Point p0, Point p1); 

};  


/* Constructors are needed if you need to reorder arguments, validate arguments, modify arguments, 
 * establish invariants */ 
/** @struct Address_w_ctor
 * @brief struct Address with custom constructor 
 * */
struct Address_w_ctor {
	std::string name; 		// "Jim Dandy" 
	int number;				// 61
	std::string street;		// "South St"
	std::string town;		// "New Providence" 
	char state[2];			// 'N' 'J' 
	char zip[5];			// 07974

	Address_w_ctor(const std::string& n, int nu, const std::string& s, 
		const std::string& t, const std::string& st, int z) ; 
};   
 


#endif // __STRUCTS_H__
