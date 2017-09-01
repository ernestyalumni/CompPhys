/** \file ptrs_unique_shared.cpp
 * \author Ernest Yeung
 * \email  ernestyalumni@gmail.com
 * \brief Simple examples for unique and shared pointers, for C++11, C++14; 
 * 			try playing around with the compilation standard flag, -std=c++XX 
 * 			where XX=11 or 14, and see how it works (or doesn't) 
 * @ref : http://en.cppreference.com/w/cpp/memory/unique_ptr
 * http://en.cppreference.com/w/cpp/memory/shared_ptr
 * 
 * If you find this code useful, feel free to donate directly and easily at this direct PayPal link: 
 * 
 * https://www.paypal.com/cgi-bin/webscr?cmd=_donations&business=ernestsaveschristmas%2bpaypal%40gmail%2ecom&lc=US&item_name=ernestyalumni&currency_code=USD&bn=PP%2dDonationsBF%3abtn_donateCC_LG%2egif%3aNonHosted 
 * 
 * which won't go through a 3rd. party such as indiegogo, kickstarter, patreon.  
 * Otherwise, I receive emails and messages on how all my (free) material on physics, math, and engineering have 
 * helped students with their studies, and I know what it's like to not have money as a student, but love physics 
 * (or math, sciences, etc.), so I am committed to keeping all my material open-source and free, whether or not 
 * sufficiently crowdfunded, under the open-source MIT license: 
 * 	feel free to copy, edit, paste, make your own versions, share, use as you wish.    
 * Peace out, never give up! -EY
 * 
 * */
 /**
 * Compilation tips
 *
 * g++ -std=c++14 ptrs_unique_shared.cpp -o ptrs_unique_shared.exe
 * g++ -std=c++11 ptrs_unique_shared.cpp -o ptrs_unique_shared.exe
 **/

#include <iostream>  
#include <memory>  // std::unique_ptr, std::shared_ptr

#include <vector>
#include <cstdio>
#include <fstream>
#include <cassert>

// needed for std::shared_ptr examples
#include <thread>
#include <chrono>
#include <mutex>

struct B {
	virtual void bar() { std::cout << "B::bar\n"; }

	/* 
	 * cf. http://en.cppreference.com/w/cpp/language/default_constructor
	 * Default constructors 
	 * Syntax : ClassName() = default;  
	 * 
	 * */
	virtual ~B() = default;
}; 

struct D : B 
{
	D() { std::cout << "D::D\n"; }
	~D() { std::cout << "D::~D\n"; }
	void bar() override { std::cout << "D::bar\n"; }
};

/*
 * --- Structs for shared_ptr example 
 * 
 * */
struct Base
{
	Base() { std::cout << "   Base::Base() \n"; }
	// Note : non-virtual destructor is OK here 
	~Base() { std::cout << "  Base::~Base()\n"; }
};  

struct Derived: public Base 
{
	Derived() { std::cout <<  "   Derived::Derived()\n"; }
	~Derived() { std::cout << "   Derived::~Derived()\n"; }
};

void thr(std::shared_ptr<Base> p) 
{
	std::this_thread::sleep_for(std::chrono::seconds(1));
	std::shared_ptr<Base> lp = p; // thread-safe, even though the 
									// shared use_count is incremented 
	{
		static std::mutex io_mutex;
		std::lock_guard<std::mutex> lk(io_mutex);
		std::cout << "local pointer in a thread:\n" 
					<< "  lp.get() = " << lp.get()
					<< ", lp.use_count() = " << lp.use_count() << '\n';
	}
}


/**
 * std::unique_ptr  
 * Defined in header <memory> 
 * template<
 * 	class T,
 * 	class Deleter = std::default_delete<T> 
 * > class unique_ptr;  
 * 
 * template < 
 * 	class T,
 * 	class Deleter 
 * > class unique_ptr<T[], Deleter>;  
 * 
 * std::unique_ptr is a smart pointer that owns and manages another object 
 * through a pointer and disposes of that object when unique_ptr goes out of scope  
 * 
 * */

// a function consuming a unique_ptr can take it by value or by rvalue reference  
std::unique_ptr<D> pass_through(std::unique_ptr<D> p) 
{
	p->bar();
	return p;
}


int main() 
{
	std::cout << "unique ownership semantics demo\n";  
	{
		auto p = std::make_unique<D>(); // p is a unique_ptr that owns a D // EY : 20170701 it should print out D::D since it's getting constructed
	
		/*
		 * std::move is used to indicate that an object t may be "moved from", i.e. 
		 * allowing efficient transfer of resources from t to another object
		 * */

		auto q = pass_through(std::move(p)); // EY:20170701 we defined pass_through function above to print out bar at least once before giving up a unique_ptr
		assert(!p); // now p owns nothing and holds a pointer
		q->bar(); 	// and q owns the D object
	} // ~D called here  EY : 20170701 so it should print out D::~D since it gets destroyed
	
	std::cout << "Runtime polymorphism demo\n";
	{
		std::unique_ptr<B> p = std::make_unique<D>(); 	// p is a unique_ptr that owns a D
														// as a pointer to base
		p->bar(); 	// virtual dispatch
		
		std::vector<std::unique_ptr<B>> v; 		// unique_ptr can be stored in a container
		v.push_back(std::make_unique<D>());
		v.push_back(std::move(p)); 
		v.emplace_back(new D);
		for (auto & p : v) { 
			p->bar(); // virtual dispatch  
		}
	} // ~D called 3 times 
	
	std::cout << "Custom deleter demo\n";  
	std::ofstream("demo.txt") << 'x' ; // prepare the file to read  
	{
		std::unique_ptr<std::FILE, decltype(&std::fclose)> fp(std::fopen("demo.txt", "r"), 
																& std::fclose); 
		if (fp) // fopen could have failed; in which case fp holds a null pointer 
			std::cout << (char) std::fgetc(fp.get()) << '\n'; 	
	}	// fclose() called here, but only if FILE* is not a null pointer
		// (that is, if fopen succeeded)  
		
	std::cout << "Custom lambda-expression deleter demo\n"; 
	{
		std::unique_ptr<D, std::function<void(D*)>> p(new D, [](D* ptr)
			{
				std::cout << "destroying from a custom deleter ... \n"; 
				delete ptr; 
			}); 	// p owns D 
		p->bar(); 
	}	// the lambda above is called and D is destroyed  
		
	std::cout << "Array form of unique_ptr demo\n"; 
	{
		std::unique_ptr<D[]> p(new D[3]);
	} 	// calls ~D 3 times 







}
