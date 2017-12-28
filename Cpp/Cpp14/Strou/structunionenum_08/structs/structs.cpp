/**
 * @file   : structs.cpp
 * @brief  : structs in separate file, in C++14, 
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
#include "./structs.h"  // current_global

/**
 * @struct Address_sep
 * @brief struct Address declaration in header, implementation/content/body separate
 * @details struct is aggregate of elements of arbitrary types 
 * @ref https://stackoverflow.com/questions/15786902/c-structs-defined-in-header-and-implemented-in-cpp
 * BIG NOTE you can't have an incomplete type; all forward declaration of struct's members and member functions 
 * must be here
 */
/* error, incomplete type, 
struct Address_sep {
	const char* name;  		// "Jim Dandy" 
	int number;				// 61 
	const char* street;		// "South St"
	const char* town;		// "New Providence"
	char state[2]; 			// 'N', 'J'
	const char* zip; 		// "07974" 
} ;
*/

/** @fn f
 * @details variables of type Address (newly defined struct) can be declared like other variables, and 
 * individual members can be accessed using . (dot) operator
 * Uses Address_hdr struct  
 * */
void f() {
	Address_hdr jd;
	jd.name = "Jim Dandy";
	jd.number = 61; 
	
}

// ERROR: Address_sep doesn't work because it's an incomplete type
/** @fn f
 * @details variables of type Address (newly defined struct) can be declared like other variables, and 
 * individual members can be accessed using . (dot) operator
 * Uses Address_sep struct  
 * */
/*void f_sep() {
	Address_sep jd;
	jd.name = "Jim Dandy";
	jd.number = 61;
}*/


/** @fn print_addr
 *  @details Structures often accessed through pointers using -> (struct pointer dereference) operator 
	 * */ 
void print_addr(Address_hdr* p) 
{
	std::cout << p->name << '\n' 
				<< p->number << ' ' << p->street << '\n' 
				<< p->town << '\n' 
				<< p->state[0] << p->state[1] << ' ' << p->zip << '\n';
}	

/** @fn print_addr2
 * @details a struct can be passed by reference and accessed using the . (struct member access) operator 
 * */
void print_addr2(const Address_hdr& r)
{
	 std::cout << r.name << '\n' 
				<< r.number << ' ' << r.street << '\n' 
				<< r.town << '\n' 
				<< r.state[0] << r.state[1] << ' ' << r.zip << '\n'; 
}

/** @fn set_current
 * @details Objects of structure types can be assigned, passed as function arguments, and 
 * returned as a result from a function.  Example: 
 * */
Address_hdr set_current(Address_hdr next) 
{
	Address_hdr prev = current_global;
	current_global = next; 
	return prev;
}

/* ========== 8.2.3 Structures and Classes ========== */
/* ===== struct is simply a class where the members are public by default.  
 * So struct can have member functions; in particular, a struct can have constructors */ 
/** @struct Points 
 * @brief constructors
 * @details Points struct, using Point struct
 * */
// constructors for Points

Points :: Points(Point p0) {
	elem.push_back(p0); 
}

Points :: Points(Point p0, Point p1) {
	elem.push_back(p0); 
	elem.push_back(p1);
}

/* Constructors are needed if you need to reorder arguments, validate arguments, modify arguments, 
 * establish invariants */ 
/** @struct Address_w_ctor
 * @brief struct Address with custom constructor 
 * */
// custom constructor for Address_w_ctor
Address_w_ctor :: Address_w_ctor(const std::string& n, int nu, const std::string& s, 
	const std::string& t, const std::string& st, int z) 
		// validate postal code
	: name{n},
	number{nu},
	street{s},
	town{t}
{
	if (st.size() != 2) 
	{
		std::cerr << "State abbreviation should be two characters" << std::endl; 
	}
	//state = {st[0] , st[1]}; 		// store postal code as characters 
	state[0] = st[0];
	state[1] = st[1];

	std::ostringstream ost; // an output string stream; see Sec. 38.4.2  
	
	ost << z; 
	std::string zi { ost.str() }; 	// extract characters from int, i.e., int -> std::string
	
	switch (zi.size()) {
		case 5:
			zip[0] = zi[0];
			zip[1] = zi[1];
			zip[2] = zi[2];
			zip[3] = zi[3];
			zip[4] = zi[4];
	
			break;
		case 4: 	// starts with '0'
			zip[0] = '0';
			zip[1] = zi[0];
			zip[2] = zi[1];
			zip[3] = zi[2];
			zip[4] = zi[3];
			break;
		default:
			std::cerr << "unexpected ZIP code format" << std::endl; 
	}
	// ... check that the code makes sense
	
}


 
 



