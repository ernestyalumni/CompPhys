//------------------------------------------------------------------------------
/// \file mycomplex.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Program 12.2. The template for the mycomplex classes.
///   string that relies on fact that a character can be copied.
/// \details The basic mechanisms for defining and using class templates are
///   introduced through the example of a string template.
/// \ref pp. 240. Edward Scheinerman.  C++ for Mathematicians: An Introduction 
///   for Students and Professionals. 2006.  Ch. 12 Polynomials.
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
///  g++ -std=c++14 FileOpen_main.cpp FileOpen.cpp -o FileOpen_main
//------------------------------------------------------------------------------
#ifndef MY_COMPLEX_H
#define MY_COMPLEX_H

#include <iostream>

template <class T>
class mycomplex
{
  private:
    T real_part;
    T imag_part;

  public:
    mycomplex<T>()
    {
      real_part = T(0);
      imag_part = T(0);
    }

    mycomplex<T>(T a)
    {
      real_part = a;
      imag_part = T(0);
    }

    mycomplex<T>(T a, T b)
    {
      real_part = a;
      imag_part = b;
    }

    T re() const { return real_part;}
    T im() const { return imag_part;}
};

template <class T>
std::ostream& operator<< (std::ostream& os, const mycomplex<T>& z)
{
  os << "(" << z.re() << ") + (" << z.im() << ")i";
  return os;
}

#endif
