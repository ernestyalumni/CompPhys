/**
 * @file   : main.cpp
 * @brief  : main driver file for struct, union, enum examples, in C++11/14, 
 * @details : struct, union, enum
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
 * g++ structunionenum.cpp -o structunionenum
 * 
 * */
#include <iostream> 
#include "./structs/structs.h"  // Point, Points




int main(int argc, char* argv[]) {
	
	/* Variables of struct types can be initialized using {} notation (Sec. 6.3.5 of Stroustrup) */
	/**
	 * @details BIG NOTE: notice how Address_hdr is completely defined in header file.  No need to 
	 * compile the structs.cpp as separate object to use it (EY : 20171227 all in stack text/code memory?   
	 * */ 
	
	Address_hdr jd_hdr = { "Jim Dandy", 61, "South St", "New Providence", {'N', 'J'}, "07975" };
	Address_hdr jd_hdr_part  {"Jim Dandy",61,"South St"}; 

	std::cout << jd_hdr.name << jd_hdr.number << jd_hdr.street << 
		jd_hdr_part.name << jd_hdr_part.number << jd_hdr_part.street << std::endl;
	
	// ERROR: incomplete type if Address_sep is separated from header file
/*	Address_sep jd_sep = {"Jim Dandy", 61, "South St", "New Providence", {'N','J'}, "07975" }; 
*/		
	
	// needs structs.cpp
	f(); 

	print_addr(&jd_hdr);
	print_addr2(jd_hdr);

	set_current(jd_hdr); 
	
	
	/** 
 * Objects of structure types can be assigned, passed as function arguments, and 
 * returned as a result from a function.  Example: 
 * @details NOTE, it compiles; but doesn't work; static Address_hdr current_global compiles, but at run-time, 
 * when applying function to assign values to it, program stops  
 * */
	//	print_addr2(current_global);
//	std::cout << current_global.name << current_global.number << std::endl;


	/* ========== 8.2.3 Structures and Classes ========== */
	/* ===== struct is simply a class where the members are public by default.  
	 * So struct can have member functions; in particular, a struct can have constructors */ 
	Point pt1 {1,2};
	Point pt2 {3,4};
	Points pts { pt1, pt2}; 
	std::cout << std::endl << pts.elem[0].x << " " << pts.elem[1].y << std::endl; 


	auto addr_inst_w_ctor = Address_w_ctor(std::string("Jim Dandy"), 61, 
		std::string("South St"), std::string("New Providence"), std::string("NJ"), 7974); 

	std::cout << addr_inst_w_ctor.name << addr_inst_w_ctor.number << 
		addr_inst_w_ctor.street << addr_inst_w_ctor.town << std::endl; 

	Point points[3] = {{1,2},{3,4},{5,6}};
	int x2 = points[2].x; 
	
/*
 * error: too many initializers for main(int, char**):: Array'  
 * 
	struct Array {
		Point elem[3]; 
	};
	
	Array points2 = {{1,2},{3,4},{5,6}}; 
	int y2 = points2.elem[2].y;
	*/
	
}
