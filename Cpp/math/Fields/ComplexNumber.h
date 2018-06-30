//------------------------------------------------------------------------------
/// \file   : ComplexNumber.h
/// \author : Ernest Yeung
/// \email  : ernestyalumni@gmail.com
/// \brief  : ComplexNumber numbers as Concrete class or Arithmetic type, parametrized 
/// \ref    : 3.2.1.1 An Arithmetic Type, Ch. 3 A Tour of C++: Abstraction 
///  	Mechanisms. Bjarne Stroustrup, The C++ Programming Language, 4th Ed.
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
///   g++ -std=c++14 ComplexNumbers_main.cpp ComplexNumbers.cpp -o ComplexNumbers_main
//------------------------------------------------------------------------------
#ifndef _COMPLEXNUMBER_H_
#define _COMPLEXNUMBER_H_

#include <iostream> // std::ostream

namespace Fields
{

template<typename T>
class ComplexNumber
{
	public:
		ComplexNumber(T r, T i);	// construct complex from 2 scalars
		ComplexNumber(T r);						// construct complex from 1 scalar
		ComplexNumber();										// default complex: {0,0}

    ComplexNumber(const ComplexNumber&) = default;
    ComplexNumber& operator=(const ComplexNumber&) = default;

    ComplexNumber(ComplexNumber&&) = default;
    ComplexNumber& operator=(ComplexNumber&&) = default;

    ~ComplexNumber() = default;

		// Accessors
		T real() const;

		T imag() const;

		// Setters
		void real(const T d);

		void imag(const T d);

		// unary arithmetic
		ComplexNumber& operator+=(const ComplexNumber& z);

		ComplexNumber& operator-=(const ComplexNumber& z);

		ComplexNumber& operator*=(const ComplexNumber&); 		// defined out-of-class somewhere
		ComplexNumber& operator/=(const ComplexNumber&);

    // binary arithmetic
    template<typename K>
    friend ComplexNumber<K> operator+(
      ComplexNumber<K> a, const ComplexNumber<K>& b);

    template<typename K>
    friend ComplexNumber<K> operator-(
      ComplexNumber<K> a, const ComplexNumber<K>& b);

    template<typename K>
    friend ComplexNumber<K> operator-(const ComplexNumber<K>& a);

    template<typename K>
    friend ComplexNumber<K> operator*(
      ComplexNumber<K> a, const ComplexNumber<K>& b);

    template<typename K>
    friend ComplexNumber<K> operator/(
      ComplexNumber<K> a, const ComplexNumber<K>& b);

		/// ComplexNumbers Conjugation
		//--------------------------------------------------------------------------
		/// \brief Return the complex conjugate of this complex number.
		//--------------------------------------------------------------------------
		ComplexNumber conjugate() const;

		//--------------------------------------------------------------------------
		/// \brief Conjugate this complex number itself.
		//--------------------------------------------------------------------------
		void conjugation();

    ComplexNumber additiveIdentity() const;
    ComplexNumber multiplicativeIdentity() const;

//    ComplexNumbers additiveInverse(const ComplexNumbers&) const;
    ComplexNumber additiveInverse() const;
//    ComplexNumbers multiplicativeInverse(const ComplexNumbers&) const;
    ComplexNumber multiplicativeInverse() const;

    template<typename K>
    friend std::ostream& operator<<(
      std::ostream& os,
      const ComplexNumber<K> & z);


	private:
		T im_;
		T re_;		

}; // class ComplexNumbers

//------------------------------------------------------------------------------
/// \details Definitions of == and != are straightforward:
//------------------------------------------------------------------------------
template<typename T>
bool operator==(const ComplexNumber<T>& a, const ComplexNumber<T>& b);		// equal

template<typename T>
bool operator!=(const ComplexNumber<T>& a, const ComplexNumber<T>& b); 	// not equal 

// originally from Stroustrup, pp. 61
//ComplexNumbers sqrt(ComplexNumbers);
//------------------------------------------------------------------------------
// \ref https://en.wikipedia.org/wiki/ComplexNumbers_number#Elementary_operations
/// \details r = |z| = |x + yi| = \sqrt{x^2 + y^2}
//------------------------------------------------------------------------------
template<typename T>
T modulus(const ComplexNumber<T>& z);

template<typename T>
T modulusSquared(const ComplexNumber<T>& z);

template<typename T>
ComplexNumber<T> additiveInverse(const ComplexNumber<T>&);

template<typename T>
ComplexNumber<T> multiplicativeInverse(const ComplexNumber<T>&);

template <typename T>
std::ostream& operator<<(std::ostream& os, const ComplexNumber<T>& z)
{
  if (z.imag() == T(0))
  {
    os << z.real();
    return os;
  }

  os << "(" << z.real() << ") + (" << z.imag() << ")i";
  return os;
}

} // namespace Fields

#endif // _COMPLEXNUMBERS_H_
