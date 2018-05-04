/**
 * @file   : ComplexNumbersGroups.h
 * @author : Ernest Yeung
 * @email  : ernestyalumni@gmail.com
 * @brief  : Complex numbers as an Abelian group 
 * @details Concrete class - defining property is its representation is its 
 * 	 definition
 * @ref    : 3.2.1.1 An Arithmetic Type, Ch. 3 A Tour of C++: Abstraction 
 * 	Mechanisms. Bjarne Stroustrup, The C++ Programming Language, 4th Ed.
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
 *  g++ -std=c++14 ../Fields/ComplexNumber.cpp ComplexNumberGroup_main.cpp -o ComplexNumberGroup_main
 * */
#ifndef _COMPLEXNUMBERSGROUP_H_
#define _COMPLEXNUMBERSGROUP_H_

#include "../Fields/ComplexNumber.h"
#include "Group.h"

namespace Groups
{

template <typename T>
class AbelianComplexNumber : public AbelianGroup<AbelianComplexNumber<T>>
{
	public:

		AbelianComplexNumber(T r, T i):
			z_{r, i}
		{}

		AbelianComplexNumber(T r):
			z_{r}
		{}

		AbelianComplexNumber():
			z_{}
		{}

		explicit AbelianComplexNumber(const Fields::ComplexNumber<T>& z):
			z_{z}
		{}

		AbelianComplexNumber(const AbelianComplexNumber&) = default;
		AbelianComplexNumber& operator=(const AbelianComplexNumber&) = default;

		AbelianComplexNumber(AbelianComplexNumber&&) = default;
		AbelianComplexNumber& operator=(AbelianComplexNumber&&) = default;

		~AbelianComplexNumber()
		{}

		// Accessors
		T real() const 
		{
			return z_.real();
		}

		T imag() const
		{
			return z_.imag();
		}

		// Setters
		void real(const T d)
		{
			z_.real(d);
		}

		void imag(const T d)
		{
			z_.imag(d);
		}

		// unary arithmetic
		AbelianComplexNumber<T>& operator+=(const AbelianComplexNumber& z)
		{
			z_ += z.z_;

			return *this;
		}

    template<typename K>
		friend AbelianComplexNumber<K> operator+(
			AbelianComplexNumber<K> z, const AbelianComplexNumber<K>& w)
		{
//			z.z_ += w.z_;
			z += w;
			return z;
		}

		AbelianComplexNumber<T> identity() const
		{
			return AbelianComplexNumber<T>{z_.additiveIdentity()};
		}

		AbelianComplexNumber<T> inverse() const
		{
			return AbelianComplexNumber<T>{z_.additiveInverse()};
		}

	private:

		Fields::ComplexNumber<T> z_;
//		Fields::ComplexNumbers<T> z_inverse_;
	//	Fields::ComplexNumbers<T> identity_{};
}; // class AbelianComplexNumbers

} // namespace Groups

#endif // _COMPLEXNUMBERSGROUP_H_
