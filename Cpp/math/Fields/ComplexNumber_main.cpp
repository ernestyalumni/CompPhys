/**
 * @file   : ComplexNumber_main.cpp
 * @author : Ernest Yeung
 * @email  : ernestyalumni@gmail.com
 * @brief  : Complex numbers as Concrete class, arithmetic type, parametrized 
 * @details Concrete class - defining property is its representation is its 
 * 	 definition
 * @ref    : 3.2.1.1 An Arithmetic Type, Ch. 3 A Tour of C++: Abstraction 
 * 	Mechanisms. Bjarne Stroustrup, The C++ Programming Language, 4th Ed.
 *  	https://stackoverflow.com/questions/8752837/undefined-reference-to-template-class-constructor
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
 *  g++ -std=c++14 ComplexNumbers_main.cpp ComplexNumbers.cpp -o ComplexNumbers_main
 * */
#include "ComplexNumber.h"

#include <iostream>

using namespace Fields;

int main()
{

	//----------------------------------------------------------------------------
	/// ComplexNumberConstructsFromTwoScalars
	ComplexNumber<double> a2{0.1, 2.3};
	std::cout << a2.real() << " " << a2.imag() << '\n';

	/// ComplexNumberConstructsFromOneScalar
	ComplexNumber<double> a1{2.3};
	std::cout << a1.real() << " " << a1.imag() << '\n';

	/// ComplexNumberConstructs
	ComplexNumber<double> a{};
	std::cout << a.real() << " " << a.imag() << '\n';

	/// MemberFunctionAdditiveIdentityReturnsZero
	ComplexNumber<double> zI{};
	std::cout << (zI.additiveIdentity()).real() << " " << 
		(zI.additiveIdentity()).imag() << '\n';

	/// MemberFunctionAdditiveInverseReturnsAdditiveInverse
	ComplexNumber<double> zInv{1., 2.};
	std::cout << (zInv.additiveInverse()).real() << " " << 
		(zInv.additiveInverse()).imag() << '\n';

	/// AdditiveInverseReturnsAdditiveInverse
	ComplexNumber<double> wInv{1., 2.};
	std::cout << (additiveInverse<double>(wInv)).real() << " " <<
		(additiveInverse<double>(wInv)).imag() << '\n';


}
