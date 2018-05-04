/**
 * @file   : ComplexNumberGroup_main.cpp
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
 *  g++ -std=c++14 ../Fields/ComplexNumber.cpp ComplexNumberGroup_main.cpp -o ComplexNumberGroup_main
 * */
#include "ComplexNumberGroup.h"

#include <iostream>

using namespace Groups;

int main()
{
	//----------------------------------------------------------------------------
	/// AbelianComplexNumberConstructsFromTwoScalars
	AbelianComplexNumber<double> abelian_z2 {2.3, 1.2};
	std::cout << abelian_z2.real() << " " << abelian_z2.imag() << '\n';

	//----------------------------------------------------------------------------
	/// AbelianComplexNumberConstructsFromOneScalars
	AbelianComplexNumber<double> abelian_z1 {2.3};
	std::cout << abelian_z1.real() << " " << abelian_z1.imag() << '\n';

	//----------------------------------------------------------------------------
	/// AbelianComplexNumberConstructs
	AbelianComplexNumber<double> abelian_z {};
	std::cout << abelian_z.real() << " " << abelian_z.imag() << '\n';

	//----------------------------------------------------------------------------
	/// AbelianComplexNumberMemberFunctionIdentityReturnsZero
	AbelianComplexNumber<double> abelian_I{32., 64.};
	std::cout << abelian_I.real() << " " << abelian_I.imag() << '\n';
	std::cout << abelian_I.identity().real() << " " << 
		abelian_I.identity().imag() << '\n';

	//----------------------------------------------------------------------------
	/// AbelianComplexNumberMemberFunctionInverseReturnsInverse
	AbelianComplexNumber<double> abelian_inverse{32., 64.};
	std::cout << abelian_inverse.inverse().real() << " " << 
		abelian_inverse.inverse().imag() << '\n';

	//----------------------------------------------------------------------------
	/// AdditionAssignmentOfAbelianComplexNumberDoesAddition
	abelian_z2 += abelian_z2;
	std::cout << abelian_z2.real() << " " << abelian_z2.imag() << '\n';

	//----------------------------------------------------------------------------
	/// AdditionAssignmentOfIdentityReturnsOriginalAbelianComplexNumber
	abelian_z2 += abelian_I.identity();
	std::cout << abelian_z2.real() << " " << abelian_z2.imag() << '\n';

	//----------------------------------------------------------------------------
	/// AdditionDoesAddition
/*
$ g++ -std=c++14 ../Fields/ComplexNumber.cpp ComplexNumberGroup_main.cpp -o ComplexNumberGroup_main
/tmp/ccIrlf8k.o: In function `Groups::AbelianComplexNumber<double>::AbelianComplexNumber(Groups::AbelianComplexNumber<double> const&)':
ComplexNumberGroup_main.cpp:(.text._ZN6Groups20AbelianComplexNumberIdEC2ERKS1_[_ZN6Groups20AbelianComplexNumberIdEC5ERKS1_]+0x1f): undefined reference to `Groups::AbelianGroup<Groups::AbelianComplexNumber<double> >::AbelianGroup(Groups::AbelianGroup<Groups::AbelianComplexNumber<double> > const&)'
/tmp/ccIrlf8k.o: In function `Groups::AbelianComplexNumber<double>::AbelianComplexNumber(Groups::AbelianComplexNumber<double>&&)':
ComplexNumberGroup_main.cpp:(.text._ZN6Groups20AbelianComplexNumberIdEC2EOS1_[_ZN6Groups20AbelianComplexNumberIdEC5EOS1_]+0x1f): undefined reference to `Groups::AbelianGroup<Groups::AbelianComplexNumber<double> >::AbelianGroup(Groups::AbelianGroup<Groups::AbelianComplexNumber<double> >&&)'
collect2: error: ld returned 1 exit status
*/

//	AbelianComplexNumber<double> addition_result {abelian_I + abelian_inverse};
//	auto addition_result = abelian_I + abelian_inverse;
}
