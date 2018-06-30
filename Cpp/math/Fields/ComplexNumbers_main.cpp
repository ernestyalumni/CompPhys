//------------------------------------------------------------------------------
/// \file   : ComplexNumbers_main.cpp
/// \author : Ernest Yeung
/// \email  : ernestyalumni@gmail.com
/// \brief  : Complex numbers as Concrete class, arithmetic type, parametrized 
/// \details Concrete class - defining property is its representation is its 
/// 	 definition
/// \ref    : 3.2.1.1 An Arithmetic Type, Ch. 3 A Tour of C++: Abstraction 
/// 	Mechanisms. Bjarne Stroustrup, The C++ Programming Language, 4th Ed.
///  	https://stackoverflow.com/questions/8752837/undefined-reference-to-template-class-constructor
/// \copyright If you find this code useful, feel free to donate directly and
/// easily at this direct PayPal link: 
///
/// https://www.paypal.com/cgi-bin/webscr?cmd=_donations&business=ernestsaveschristmas%2bpaypal%40gmail%2ecom&lc=US&item_name=ernestyalumni&currency_code=USD&bn=PP%2dDonationsBF%3abtn_donateCC_LG%2egif%3aNonHosted 
/// 
/// which won't go through a 3rd. party such as indiegogo, kickstarter, patreon.  
/// Otherwise, I receive emails and messages on how all my (free) material on
/// physics, math, and engineering have helped students with their studies, and
/// I know what it's like to not have money as a student, but love physics (or 
/// math, sciences, etc.), so I am committed to keeping all my material 
/// open-source and free, whether or not sufficiently crowdfunded, under the 
/// open-source MIT license: feel free to copy, edit, paste, make your own
/// versions, share, use as you wish.    
/// Peace out, never give up! -EY
//------------------------------------------------------------------------------
/// COMPILATION TIPS:
/// 	g++ -std=c++14 ComplexNumbers_main.cpp ComplexNumbers.cpp -o ComplexNumbers_main
//------------------------------------------------------------------------------
#include "ComplexNumbers.h"

#include <limits>
#include <iostream>

using namespace Fields;

int main()
{

	//----------------------------------------------------------------------------
	/// ComplexNumbersConstructsFromTwoScalars
	ComplexNumbers<double> a2{0.1, 2.3};
	std::cout << a2.real() << " " << a2.imag() << '\n';

	/// ComplexNumbersConstructsFromOneScalar
	ComplexNumbers<double> a1{2.3};
	std::cout << a1.real() << " " << a1.imag() << '\n';

	/// ComplexNumbersConstructs
	ComplexNumbers<double> a{};
	std::cout << a.real() << " " << a.imag() << '\n';

	/// ComplexNumbersConstructsForNominalValuesNumericLimitsAndEpsilons
	std::cout << 
		"\n ComplexNumbersConstructsForNominalValuesNumericLimitsAndEpsilons \n";
	{
		ComplexNumbers<float> a_max {std::numeric_limits<float>::max(),
			std::numeric_limits<float>::max()};
		std::cout << a_max.real() << " " << a_max.imag() << '\n';

		ComplexNumbers<float> a_lowest {std::numeric_limits<float>::lowest(),
			std::numeric_limits<float>::lowest()};
		std::cout << a_lowest.real() << " " << a_lowest.imag() << '\n';

		ComplexNumbers<float> a_epsilon {std::numeric_limits<float>::epsilon(),
			std::numeric_limits<float>::epsilon()};
		std::cout << a_epsilon.real() << " " << a_epsilon.imag() << '\n';

		ComplexNumbers<float> a_roundoff_error {
			std::numeric_limits<float>::round_error(),
			std::numeric_limits<float>::round_error()};
		std::cout << a_roundoff_error.real() << " " << a_roundoff_error.imag() << '\n';

		ComplexNumbers<double> b_max {std::numeric_limits<double>::max(),
			std::numeric_limits<double>::max()};
		std::cout << b_max.real() << " " << b_max.imag() << '\n';

		ComplexNumbers<double> b_lowest {std::numeric_limits<double>::lowest(),
			std::numeric_limits<double>::lowest()};
		std::cout << b_lowest.real() << " " << b_lowest.imag() << '\n';

		ComplexNumbers<double> b_epsilon {std::numeric_limits<double>::epsilon(),
			std::numeric_limits<double>::epsilon()};
		std::cout << b_epsilon.real() << " " << b_epsilon.imag() << '\n';

		ComplexNumbers<double> b_roundoff_error {
			std::numeric_limits<double>::round_error(),
			std::numeric_limits<double>::round_error()};
		std::cout << b_roundoff_error.real() << " " << b_roundoff_error.imag() << '\n';
	}

	/// MemberFunctionAdditiveIdentityReturnsZero
	ComplexNumbers<double> zI{};
	std::cout << (zI.additiveIdentity()).real() << " " << 
		(zI.additiveIdentity()).imag() << '\n';

	/// MemberFunctionAdditiveInverseReturnsAdditiveInverse
	ComplexNumbers<double> zInv{1., 2.};
	std::cout << (zInv.additiveInverse()).real() << " " << 
		(zInv.additiveInverse()).imag() << '\n';

	/// AdditiveInverseReturnsAdditiveInverse
	ComplexNumbers<double> wInv{1., 2.};
	std::cout << (additiveInverse<double>(wInv)).real() << " " <<
		(additiveInverse<double>(wInv)).imag() << '\n';


}
