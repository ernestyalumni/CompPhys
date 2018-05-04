/**
 * @file   : Groups.h
 * @author : Ernest Yeung
 * @email  : ernestyalumni@gmail.com
 * @brief  : Groups as an abstract type.   
 * @details A group 
 * @ref    : 3.2.1.2 A Container, Ch. 3 A Tour of C++: Abstraction 
 * 	Mechanisms. Bjarne Stroustrup, The C++ Programming Language, 4th Ed.
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
 *  feel free to copy, edit, paste, make your own versions, share, use as you wish.    
 * Peace out, never give up! -EY
 * 
 * COMPILATION TIPS:
 *  g++ -std=c++14 Groups.cpp Groups_main.cpp -o Groups_main
 * */
#ifndef _GROUPS_H_
#define _GROUPS_H_

namespace Groups
{

//------------------------------------------------------------------------------
/// \brief Group
/// \details Use CRTP pattern.
/// \ref https://stackoverflow.com/questions/27180342/pure-virtual-function-in-abstract-class-with-return-type-of-base-derived-type
//------------------------------------------------------------------------------
template <class Object>
class Group
{
	public:

		Group() = default;

		Group(const Group&) = delete;
		Group& operator=(const Group&) = delete;

		Group(Group&&) = delete;
		Group& operator=(Group&&) = delete;

		virtual Object& operator*=(const Object& g) = 0; // pure virtual function

		template <class Obj>
		friend Obj operator*(Obj g, const Obj& h);

		virtual Object identity() const = 0; // pure virtual function 
		virtual Object inverse() const = 0; // pure virtual function
	
		virtual ~Group()
		{}
};

template <class Object>
class AbelianGroup
{
	public:

		AbelianGroup() = default;

		// copy constructor needed for binary addition
		AbelianGroup(const AbelianGroup&); 
		AbelianGroup& operator=(const AbelianGroup&) = delete;

		// move constructor needed because in derived classes, identity will need it
		AbelianGroup(AbelianGroup&&);
		AbelianGroup& operator=(AbelianGroup&&) = delete;

		virtual Object& operator+=(const Object& g) = 0; 

		template <class Obj>
		friend Obj operator+(Obj g, const Obj& h);

		virtual Object identity() const = 0; // pure virtual function 
		virtual Object inverse() const = 0;
		virtual ~AbelianGroup()
		{}
};

} // namespace Groups

#endif // _GROUPS_H_
