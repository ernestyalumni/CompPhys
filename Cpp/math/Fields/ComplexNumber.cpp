//------------------------------------------------------------------------------
/// \file   : ComplexNumber.cpp
/// \author : Ernest Yeung
/// \email  : ernestyalumni@gmail.com
/// \brief  : Complex numbers as Concrete class, arithmetic type, parametrized. 
/// \ref    : 3.2.1.1 An Arithmetic Type, Ch. 3 A Tour of C++: Abstraction 
/// 	Mechanisms. Bjarne Stroustrup, The C++ Programming Language, 4th Ed.
/// \details : Using RAII for Concrete classes. 
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
/// 	g++ -std=c++14 ComplexNumber_main.cpp ComplexNumber.cpp -o ComplexNumber_main
//------------------------------------------------------------------------------
#include "ComplexNumber.h"

#include <cassert> 	// assert

#include <cmath> // std::sqrt

namespace Fields
{

template<typename T>
ComplexNumber<T>::ComplexNumber(T r, T i):
	re_{r}, im_{i}
{}

template<typename T>
ComplexNumber<T>::ComplexNumber(T r):						// construct complex from 1 scalar
	re_{r}, im_{0}
{}

template<typename T>
ComplexNumber<T>::ComplexNumber():										// default complex: {0,0}
	re_{0}, im_{0}		
{}		

// cannot be overloaded
// template<typename T>
// constexpr ComplexNumber<T>::ComplexNumber(T r = 0, T i = 0):
// 	re_{r}, im_{i}
// {}

// Accessors
template<typename T>
T ComplexNumber<T>::real() const 
{
	return re_;
}

template<typename T>
T ComplexNumber<T>::imag() const
{
	return im_;
}

// Setters
template<typename T>
void ComplexNumber<T>::real(const T d)
{
	re_ = d;
}

template<typename T>
void ComplexNumber<T>::imag(const T d)
{
	im_ = d;
}

// unary arithmetic
template<typename T>
inline ComplexNumber<T>& ComplexNumber<T>::operator+=(const ComplexNumber& z) 
{
	re_ += z.real();
	im_ += z.imag();
	return *this;
}

template<typename T>
inline ComplexNumber<T>& ComplexNumber<T>::operator-=(const ComplexNumber& z)
{
	re_ -= z.real();
	im_ -= z.imag();
	return *this;
}

//------------------------------------------------------------------------------
/// \brief Complex multiplication and division
//------------------------------------------------------------------------------
/// unary multiplication and division

//------------------------------------------------------------------------------
/// \details z*w = (a+bi)*(c+di) = (a*c - b*d) + i(a*d + b*c)
//------------------------------------------------------------------------------
template<typename T>
ComplexNumber<T>& ComplexNumber<T>::operator*=(const ComplexNumber& z)
{
	this->re_ = this->real() * z.real() - this->imag() * z.imag();
	this->im_ = this->real() * z.imag() + this->imag() * z.real();
	return *this;
}

//------------------------------------------------------------------------------
/// \details z/w = (a+bi)/(c+di) = (a+bi)(c-di)/((c+di)(c-di)) = 
/// 	((ac+bd) +i(bc-ad))/(c^2 + d^2)
//------------------------------------------------------------------------------
template<typename T>
ComplexNumber<T>& ComplexNumber<T>::operator/=(const ComplexNumber& z)
{
	assert(modulusSquared<T>(z) != 0.);
 
	this->re_ = (this->real() * z.real() + this->imag() * z.imag())/
		modulusSquared<T>(z);

	this->im_ = (-this->real() * z.imag() + this->imag() * z.real())/
		modulusSquared<T>(z);
	return *this;
}

//------------------------------------------------------------------------------
/// \brief Binary arithmetic operators
/// \ref http://en.cppreference.com/w/cpp/language/operators
/// \details Many useful operations don't require direct access to 
/// 	representation of complex, so they can be defined separately from class 
/// 	definitions.
///		Be careful about l-value references vs. r-value references 
//------------------------------------------------------------------------------
/*template<typename T>
ComplexNumber<T> ComplexNumber<T>::operator+(
	const ComplexNumber<T>& a, const ComplexNumber<T>& b)
{
	return a += b;
}*/

template<typename T>
ComplexNumber<T> operator+(
	ComplexNumber<T> a, const ComplexNumber<T>& b)
{
	return a += b; // return result by value (uses move constructor)
}

/*template<typename T>
ComplexNumber<T> ComplexNumber<T>::operator-(
	const ComplexNumber<T>& a, const ComplexNumber<T>& b)
{
	return a -= b;
}*/
template<typename T>
ComplexNumber<T> operator-(
	ComplexNumber<T> a, const ComplexNumber<T>& b)
{
	return a -= b; // return result by value (uses move constructor)
}

/*template<typename T>
ComplexNumber<T> ComplexNumber<T>::operator-(const ComplexNumber<T>& a)
{
	return {-a.real(), -a.imag()};	// unary minus
}*/
template<typename T>
ComplexNumber<T> operator-(const ComplexNumber<T>& a)
{
	return {-a.real(), -a.imag()};	// unary minus
}

/*template<typename T>
ComplexNumber<T> ComplexNumber<T>::operator*(
	const ComplexNumber<T>& a, const ComplexNumber<T>& b)
{
	return a *= b;
}*/

template<typename T>
ComplexNumber<T> operator*(
	ComplexNumber<T> a, const ComplexNumber<T>& b)
{
	return a *= b; // return result by value (uses move constructor)
}

/*template<typename T>
ComplexNumber<T> ComplexNumber<T>::operator/(
	const ComplexNumber& a, const ComplexNumber& b)
{
	return a /= b;
}*/

template<typename T>
ComplexNumber<T> operator/(
	ComplexNumber<T> a, const ComplexNumber<T>& b)
{
	return a /= b; // return result by value (uses move constructor)
}

/// Complex Conjugation
//------------------------------------------------------------------------------
/// \brief Return the complex conjugate of this complex number.
//------------------------------------------------------------------------------
template<typename T>
ComplexNumber<T> ComplexNumber<T>::conjugate() const
{
	return {re_, -im_};
}

//------------------------------------------------------------------------------
/// \brief Conjugate this complex number itself.
//------------------------------------------------------------------------------
template<typename T>
void ComplexNumber<T>::conjugation()
{
	im_ = -im_;
}

//------------------------------------------------------------------------------
/// \details Definitions of == and != are straightforward:
//------------------------------------------------------------------------------
template<typename T>
bool operator==(ComplexNumber<T>& a, ComplexNumber<T>& b)		// equal
{
	return a.real() == b.real() && a.imag() == b.imag();
}

template<typename T>
bool operator!=(ComplexNumber<T>& a, ComplexNumber<T>& b) 	// not equal 
{
	return !(a == b);
}

template<typename T>
ComplexNumber<T> ComplexNumber<T>::additiveIdentity() const
{
	return {};
}

template<typename T>
ComplexNumber<T> ComplexNumber<T>::multiplicativeIdentity() const
{
	return {1};
}

/*template<typename T>
ComplexNumber<T> ComplexNumber<T>::additiveInverse(const ComplexNumber& z) const
{
	return -z;
}*/
template<typename T>
ComplexNumber<T> ComplexNumber<T>::additiveInverse() const
{
	return -(*this);
}

/*template<typename T>
ComplexNumber<T> ComplexNumber<T>::multiplicativeInverse(
	const ComplexNumber& z) const
{
	return (ComplexNumber<T>(static_cast<T>(1)) / z);
}*/

template<typename T>
ComplexNumber<T> ComplexNumber<T>::multiplicativeInverse() const
{
	assert((real() != 0) && (imag() != 0));
	return (ComplexNumber<T>(static_cast<T>(1)) / *this);
}

//------------------------------------------------------------------------------
/// \details Definitions of == and != are straightforward:
//------------------------------------------------------------------------------
template<typename T>
bool operator==(const ComplexNumber<T>& a, const ComplexNumber<T>& b)		// equal
{
	return a.real() == b.real() && a.imag() == b.imag();
}

template<typename T>
bool operator!=(const ComplexNumber<T>& a, const ComplexNumber<T>& b) 	// not equal 
{
	return !(a==b);
}

// originally from Stroustrup, pp. 61
//Complex sqrt(Complex);
//------------------------------------------------------------------------------
// \ref https://en.wikipedia.org/wiki/Complex_number#Elementary_operations
/// \details r = |z| = |x + yi| = \sqrt{x^2 + y^2}
//------------------------------------------------------------------------------
template<typename T>
T modulus(const ComplexNumber<T>& z)
{
	return std::sqrt(z.real() * z.real() + z.imag() * z.imag());
}

template<typename T>
T modulusSquared(const ComplexNumber<T>& z)
{
	return z.real() * z.real() + z.imag() * z.imag();
}

template<typename T>
ComplexNumber<T> additiveInverse(const ComplexNumber<T>& z)
{
	return -z;
}

template<typename T>
ComplexNumber<T> multiplicativeInverse(const ComplexNumber<T>& z)
{
	return (ComplexNumber<T>(static_cast<T>(1)) / z);
}

//------------------------------------------------------------------------------
/// \brief explicit instantiations
/// \ref https://stackoverflow.com/questions/495021/why-can-templates-only-be-implemented-in-the-header-file
/// 	https://stackoverflow.com/questions/8752837/undefined-reference-to-template-class-constructor
/// 	https://stackoverflow.com/questions/4933056/how-do-i-explicitly-instantiate-a-template-function
/// \details With class templates, all implementations must be in source files 
/// 	or explicit instantiations; drawback for explicit instantiations is ALL
/// 	use cases must be first explicitly instantiations
//------------------------------------------------------------------------------
template class ComplexNumber<int>;
template class ComplexNumber<long>;
template class ComplexNumber<float>;
template class ComplexNumber<double>;

template ComplexNumber<double> additiveInverse<double>(
	const ComplexNumber<double>& z);

} // namespace Fields
