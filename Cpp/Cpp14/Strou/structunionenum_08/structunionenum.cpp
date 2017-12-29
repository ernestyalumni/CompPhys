/**
 * @file   : structunionenum.cpp
 * @brief  : struct, union, enum, in C++14, 
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

#include <iostream> // for sanity checks, std::cout  
#include <vector> // std::vector , for example of data member in struct

#include <string>  // std::string, for example of Address_w_ctor 
#include <sstream> // ostringstream

#include <array> // std::array 

/* ========== defining structs in global scope, outside of main ========== */
/** @struct Address_global
 * @brief example of a struct in global scope 
 * @details struct is an aggregate of elements of arbitrary types 
 * */
struct Address_global { 
	const char* name; 	// "Jim Dandy"
	int number; 		// 61 
	const char* street; // "South St"
	const char* town;	// "New Providence"
	char state[2]; 		// 'N' 'J' 
	const char* zip;	// "07974"	
};

Address_global jd_global1 { "Jim Dandy", 61,"South St", "New Providence",{'N','J'},"07974"};

/** @fn f
 * @brief variables of types Address (e.g. Address_global) can be declared like other variables, 
 * and individual members can be accessed using . (dot) operator 
 */
void f() 
{
	Address_global jd;
	jd.name = "Jim Dandy";
	jd.number = 61;	
}; 

/** @fn print_addr
 *  @details Structures often accessed through pointers using -> (struct pointer dereference) operator 
	 * */ 
void print_addr(Address_global* p) 
{
	std::cout << p->name << '\n' 
				<< p->number << ' ' << p->street << '\n' 
				<< p->town << '\n' 
				<< p->state[0] << p->state[1] << ' ' << p->zip << '\n';
}	

/** @fn print_addr2
 * @details a struct can be passed by reference and accessed using the . (struct member access) operator 
 * */
void print_addr2(const Address_global& r)
{
	 std::cout << r.name << '\n' 
				<< r.number << ' ' << r.street << '\n' 
				<< r.town << '\n' 
				<< r.state[0] << r.state[1] << ' ' << r.zip << '\n'; 
}

/** 
 * Objects of structure types can be assigned, passed as function arguments, and 
 * returned as a result from a function.  Example: 
 * */
Address_global current_global;

Address_global set_current(Address_global next) 
{
	Address_global prev= current_global;
	current_global = next; 
	return prev;
}

/** @struct Readout 
 * @details Members are allocated in memory in declaration order, 
 * so address of hour must be less than the address of value 
 */
struct Readout {
	char hour;		// [0:23] 
	int value;		
	char seq; 		// sequence mark['a';'z'] 
};

/** @struct Readout1 
 * @details minimize wasted space by simply ordering members by size (largest member 1st).  
 */
struct Readout1 {
	int value;		
	char hour;		// [0:23] 
	char seq; 		// sequence mark['a';'z'] 
};

/** 
 * @details name of type becomes available for use immediately after it's been encountered, 
 * not just after complete declaration 
 * */
struct Link {
	Link* previous;
	Link* successor; 
};

/* ========== 8.2.3 Structures and Classes ========== */
/* ===== struct is simply a class where the members are public by default.  
 * So struct can have member functions; in particular, a struct can have constructors */ 
/** @struct Point_global
 * @brief Point struct in the global function 
 * @details contrary to pp. 206, Sec. 8.2.3. 
 * "You do not need to define a constructor simply to initialize members in order" and 
 * "The name of a struct can be used before the type is defined as long as 
 * that use does not require name of a member or size of structure to be known" pp. 205
 * Obtain ERROR has incomplete type for a forward declaration 
 * */
struct Point_global {
	int x,y; 
};

/** @struct Point_global 
 * @brief Point struct in global scope 
 * */
struct Points_global {
	std::vector<Point_global> elem;	// must contain at least 1 Point_main
	Points_global(Point_global p0) { elem.push_back(p0); }
	Points_global(Point_global p0, Point_global p1) { 
		elem.push_back(p0); elem.push_back(p1); }
	// ...
};  

/* Constructors are needed if you need to reorder arguments, validate arguments, modify arguments, 
 * establish invariants */ 
/** @struct Address_w_ctor
 * @brief struct Address with custom constructor 
 * */
struct Address_w_ctor {
	std::string name; 		// "Jim Dandy"  
	int number;				// 61 
	std::string street; 	// "South St" 
	std::string town;		// "New Providence"  
	char state[2]; 			// 'N' 'J' 
	char zip[5];			// 07974 
	
	Address_w_ctor(const std::string& n, int nu, const std::string& s, 
		const std::string& t, const std::string& st, int z); 
}; 

// custom constructor for Address_w_ctor
Address_w_ctor::Address_w_ctor(const std::string& n, int nu, const std::string& s, 
	const std::string& t, const std::string& st, int z) 
	// validate postal code
	: name {n}, 
		number{nu}, 
		street{s}, 
		town{t}
{
	if (st.size() != 2) 
	{
//		error("State abbreviation should be two characters") // error is custom from Stroustrup's book
		std::cerr << "State abbreviation should be two characters" << std::endl;
	}
	state[0] = st[0]; 
	state[1] = st[1];

//	state = {st[0],st[1]}; 	// store postal code as characters after check

	std::ostringstream ost; // an output string stream; see Sec. 38.4.2  
	ost << z; 
	std::string zi { ost.str() }; // extract characters from int, i.e. int -> std::string
	
	switch (zi.size()) {
		case 5: 
//			zip = { zi[0], zi[1], zi[2], zi[3], zip[4]};
			zip[0] = zi[0]; 
			zip[1] = zi[1]; 
			zip[2] = zi[2]; 
			zip[3] = zi[3]; 
			zip[4] = zi[4]; 

			break; 
		case 4:		// starts with '0' 
//			zip = {'0', zi[0], zi[1], zi[2], zi[3]};
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

struct Array_global {
	Point_global elem[3];
};

Array_global shift_global(Array_global a, Point_global p) 
{
	for (int i=0; i != 3; ++i) {
		a.elem[i].x += p.x; 
		a.elem[i].y += p.y; 
	}
	return a; 
};

/* ============ using std::array instead of struct ============ */

using Array_std3 = std::array<Point_global, 3> ; // array of 3 Point_global's

Array_std3 shift_std3(Array_std3 a, Point_global p) 
{
	for (int i =0; i!=a.size(); ++i ) {
		a[i].x += p.x; 
		a[i].y += p.y;
	}
	return a;
}

/* main advantage of std::array over a built-in array are that 
 * it's a proper object type (has assignment, etc.) and 
 * doesn't implicitly convert to a pointer to an individual element 
 * */ 
std::ostream& operator << (std::ostream& os, Point_global p)
{	
	std::cout << '{' << p.x << ',' << p.y << '}';  
}

void print(Point_global a[], int s) // must specify number of elements 
{
	for (int i=0; i != s; ++i) {
		std::cout << a[i] << '\n'; 
	}
}

template<typename T, int N>
void print(std::array<T,N>& a)
{
	for (int i=0; i != a.size(); ++i) {
		std::cout << a[i] << '\n'; 
	}
}


/* =============== MAIN =============== */

int main(int argc, char* argv[]) {
	/* ========== defining structs inside main ========== */
	/** @struct Address_main
	 * @brief example of struct in the main
	 * @details struct is an aggregate of elements of arbitrary types 
	 * */
	struct Address_main {
		const char* name; 	// "Jim Dandy"
		int number;			// 61
		const char* street;	// "South St"
		const char* town;	// "New Providence"
		char state[2];		// 'N' 'J'
		const char* zip;	// "07974"  
	}; 
	
	f();

	/* Variables of struct types can be initialized using {} notation (Sec. 6.3.5 of Stroustrup) */
	Address_global jd_global = { 
		"Jim Dandy", 
		61, "South St", 
		"New Providence", 
		{'N','J'}, "07974"
	};
	
	Address_main jd_main = { "Jim Dandy", 61, "South St", "New Providence", {'N','J'}, "07974" };

	Address_global jd_global_part { "Jim Dandy", 61, "South St" }; 

	std::cout << jd_global.name << jd_global.number << jd_main.name << jd_main.number << 
		jd_global_part.name << jd_global_part.number << std::endl;

	/* try initializing with no =, only {} */ 

	Address_global jd_global1 { "Jim Dandy", 61, "South St", "New Providence", {'N','J'}, "07974"};
	
	Address_main jd_main1 { "Jim Dandy", 61, "South St", "New Providence", {'N','J'}, "07974" };

	Address_global jd_global_part1 { "Jim Dandy", 61, "South St" }; 

	std::cout << jd_global1.name << jd_global1.number << jd_main1.name << jd_main1.number << 
		jd_global_part1.name << jd_global_part1.number << std::endl;

	
	print_addr(&jd_global);
	// struct can be passed by reference
	print_addr2(jd_global);

	print_addr(&jd_global1);
	print_addr2(jd_global1);

	set_current(jd_global1);
	print_addr(&current_global);

	Readout readout_inst { '0', 5, 'a' }; 
	Readout1 readout1_inst { '1', 6, 'z' }; 

	std::cout << sizeof('0') << " " << sizeof(4) << std::endl;
	std::cout << sizeof(readout_inst) << " " << sizeof(readout1_inst) << std::endl;

	
	struct Link1 {
		Link1* previous;
		Link1* successor; 
	};

	/** @struct No_good
	 * @details Not possible to declare new objects of a struct until 
	 * its complete declaration	has been seen
	*/ 
	/*
	struct No_good {
		No_good member; // error: recursive definition 
	};
	*/
	
	/* ========== 8.2.3 Structures and Classes ========== */
	/* ===== struct is simply a class where the members are public by default.  
	 * So struct can have member functions; in particular, a struct can have constructors */ 
	/** @struct Point_main 
	 * @details Point struct in the main function 
	 * */
	struct Point_main { 
		int x,y; 
	};

	/** @struct Point_main 
	 * @details Point struct in the main function 
	 * */
	struct Points_main {
		std::vector<Point_main> elem;	// must contain at least 1 Point_main
		Points_main(Point_main p0) { elem.push_back(p0); }
		Points_main(Point_main p0, Point_main p1) { 
			elem.push_back(p0); elem.push_back(p1); }
		// ...
	};  

	Point_global pt_global1 { 1,2}; 
	Point_global pt_global2 { 3,4}; 
	Points_global pts_global { pt_global1, pt_global2 }; 

	std::cout << std::endl << pts_global.elem[0].x << " " << pts_global.elem[1].y << std::endl; 
		
	Point_main pt_main1 { 1,2}; 
	Point_main pt_main2 { 3,4}; 
	Points_main pts_main { pt_main1, pt_main2 }; 

	std::cout << pts_main.elem[0].x << " " << pts_main.elem[1].y << std::endl; 

	Point_global p0_global; 	// danger: uninitialized if in local scope (Sec. 6.3.5.1)
	Point_global p1_global {}; 	// default construction: {{},{}}; that is {0,0} 
	Point_global p2_global {1};	// the second member is default constructed: {1,{}}; that is {1,0}
	Point_global p3_global {1,2}; // {1,2}  

	Point_main p0_main; 	// danger: uninitialized if in local scope (Sec. 6.3.5.1)
	Point_main p1_main {}; 	// default construction: {{},{}}; that is {0,0} 
	Point_main p2_main {1};	// the second member is default constructed: {1,{}}; that is {1,0}
	Point_main p3_main {1,2}; // {1,2}  

	auto addr_inst_ctor = Address_w_ctor( std::string("Jim Dandy"), 61, std::string("South St"), 
		std::string("New Providence"), std::string("NJ"), 7974);

	std::cout << std::endl << addr_inst_ctor.name << addr_inst_ctor.number 
		<< addr_inst_ctor.street << addr_inst_ctor.town << std::endl; 

	/* =============== Sec. 8.2.4 Structures and Arrays =============== */


	Point_global pts_arr[3] {{1,2}, {3,4}, {5,6}}; 
	int x2 = pts_arr[2].x; 
	

	struct Array_main {
		Point_main elem[3];
	};

//	Array_global points2_global =  {{1,2},{3,4},{5,6}};  // error: too many initializers for 'Array_global'
//	int y2 = points2.elem[2].y;

	Array_global points2_global;
	points2_global.elem[0] = Point_global({1,2});
	points2_global.elem[1] = Point_global({3,4});
	points2_global.elem[2] = Point_global({5,6});

//	Array_global ax_global = shift_global()
	Array_global ax_global = shift_global(points2_global, Point_global({10,20}) );

	std::cout << ax_global.elem[0].x << " " << ax_global.elem[0].y << " " 
		<< ax_global.elem[1].x << " " << ax_global.elem[1].y << " "  
		<< ax_global.elem[2].x << " " << ax_global.elem[2].y << std::endl;

// need the following line in global scope for function shift2
//	using Array_std3 = std::array<Point_global, 3> ; // array of 3 Point_global's

//	Array_std3 pts3  ={ {1,2}, {3,4}, {5,6}}; // error: too many initializers for 
	// ‘Array_std3 {aka std::array<Point_global2, 3ul>}’

	Array_std3 pts3  ={Point_global{1,2}, Point_global{3,4}, Point_global{5,6}};

	Array_std3 ax3 = shift_std3(pts3, Point_global({10,20}));
	
	std::cout << ax3[0].x << " " << ax3[0].y << " " 
		<< ax3[1].x << " " << ax3[1].y << " "  
		<< ax3[2].x << " " << ax3[2].y << std::endl;


	print(pts_arr,3);  // {1,2} \\ {3,4} \\ {5,6} 
	auto points2 = std::array<Point_global,3>( {Point_global{1,2}, Point_global{3,4}, Point_global{5,6}});
	print<Point_global,3>(points2 ) ; 

	/** @ref pp. 209 Ch.8, Sec. 8.2.4, of Stroustrup
	 * @details "Disadvantage of std::array compared to a built-in array is that 
	 * we can't deduce the number of elements from the length of the initializer" 
	 * */ 
	
	// 8.2.5 Type Equivalence  
	// 2 structs are different types even when they have the same members.  
	
	// 8.2.6. Plain Old Data 
	
	
	struct S0 {};	// a POD
	struct S1 {int a; }; // a POD
	struct S2 {
		int a;
		S2(int aa) : a(aa) {} };	// not a POD (no default constructor)  
	struct S3 { 
		int a;
		S3(int aa) : a(aa) { }  
		S3() {} }; // a POD (user-defined default constructor  
	struct S4 { int a; 
		S4(int aa) : a(aa) { }  
		S4() = default; }; // a POD
//	struct S5 { virtual void f(); /* ... */ }; // not a POD (has a virtual function)  	
	
	struct S6 : S1 {}; 		// a POD
	struct S7 : S0 { int b; }; // a POD  
	struct S8 : S1 { int b; }; // not a POD (data in both S1 and S8) 
	struct S9 : S0, S1 {} ; // a POD
	
		
	 
	



}
