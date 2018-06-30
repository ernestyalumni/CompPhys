//------------------------------------------------------------------------------
/// \file   : ComplexNumbers.h
/// \author : Ernest Yeung
/// \email  : ernestyalumni@gmail.com
/// \brief  : ComplexNumber numbers as Concrete class or Arithmetic type, parametrized 
/// \ref    : 3.2.1.1 An Arithmetic Type, Ch. 3 A Tour of C++: Abstraction 
///   Mechanisms. Bjarne Stroustrup, The C++ Programming Language, 4th Ed.
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
///   g++ -std=c++14 ComplexNumber_main.cpp ComplexNumber.cpp -o ComplexNumber_main
//------------------------------------------------------------------------------
#ifndef _COMPLEXNUMBERS_H_
#define _COMPLEXNUMBERS_H_

#include <iostream> // std::ostream

namespace Fields
{

template<typename T>
class ComplexNumbers
{
	public:
		ComplexNumbers(T r, T i);	// construct complex from 2 scalars
		ComplexNumbers(T r);						// construct complex from 1 scalar
		ComplexNumbers();										// default complex: {0,0}

    ComplexNumbers(const ComplexNumbers&) = default;
    ComplexNumbers& operator=(const ComplexNumbers&) = default;

    ComplexNumbers(ComplexNumbers&&) = default;
    ComplexNumbers& operator=(ComplexNumbers&&) = default;

    ~ComplexNumbers() = default;

		// Accessors
		T real() const;

		T imag() const;

		// Setters
		void real(const T d);

		void imag(const T d);

		// unary arithmetic
		ComplexNumbers& operator+=(const ComplexNumbers& z);

		ComplexNumbers& operator-=(const ComplexNumbers& z);

		ComplexNumbers& operator*=(const ComplexNumbers&); 		// defined out-of-class somewhere
		ComplexNumbers& operator/=(const ComplexNumbers&);

    // binary arithmetic
    template<typename K>
    friend ComplexNumbers<K> operator+(
      ComplexNumbers<K> a, const ComplexNumbers<K>& b);

    template<typename K>
    friend ComplexNumbers<K> operator-(
      ComplexNumbers<K> a, const ComplexNumbers<K>& b);

    template<typename K>
    friend ComplexNumbers<K> operator-(const ComplexNumbers<K>& a);

    template<typename K>
    friend ComplexNumbers<K> operator*(
      ComplexNumbers<K> a, const ComplexNumbers<K>& b);

    template<typename K>
    friend ComplexNumbers<K> operator/(
      ComplexNumbers<K> a, const ComplexNumbers<K>& b);

		/// ComplexNumbers Conjugation
		//--------------------------------------------------------------------------
		/// \brief Return the complex conjugate of this complex number.
		//--------------------------------------------------------------------------
		ComplexNumbers conjugate() const;

		//--------------------------------------------------------------------------
		/// \brief Conjugate this complex number itself.
		//--------------------------------------------------------------------------
		void conjugation();

    ComplexNumbers additiveIdentity() const;
    ComplexNumbers multiplicativeIdentity() const;

//    ComplexNumbers additiveInverse(const ComplexNumbers&) const;
    ComplexNumbers additiveInverse() const;
//    ComplexNumbers multiplicativeInverse(const ComplexNumbers&) const;
    ComplexNumbers multiplicativeInverse() const;

    template<typename K>
    friend std::ostream& operator<<(
      std::ostream& os,
      const ComplexNumbers<K> & z);

	private:
		T im_;
		T re_;		

}; // class ComplexNumbers

//------------------------------------------------------------------------------
/// \details Definitions of == and != are straightforward:
//------------------------------------------------------------------------------
template<typename T>
bool operator==(const ComplexNumbers<T>& a, const ComplexNumbers<T>& b);		// equal

template<typename T>
bool operator!=(const ComplexNumbers<T>& a, const ComplexNumbers<T>& b); 	// not equal 

// originally from Stroustrup, pp. 61
//ComplexNumbers sqrt(ComplexNumbers);
//------------------------------------------------------------------------------
// \ref https://en.wikipedia.org/wiki/ComplexNumbers_number#Elementary_operations
/// \details r = |z| = |x + yi| = \sqrt{x^2 + y^2}
//------------------------------------------------------------------------------
template<typename T>
T modulus(const ComplexNumbers<T>& z);

template<typename T>
T modulusSquared(const ComplexNumbers<T>& z);

template<typename T>
ComplexNumbers<T> additiveInverse(const ComplexNumbers<T>&);

template<typename T>
ComplexNumbers<T> multiplicativeInverse(const ComplexNumbers<T>&);

template <typename T>
std::ostream& operator<<(std::ostream& os, const ComplexNumbers<T>& z)
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
