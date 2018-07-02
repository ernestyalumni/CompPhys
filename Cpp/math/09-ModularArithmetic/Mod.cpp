//------------------------------------------------------------------------------
/// \file Mod.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Program 9.2. Source file for the Mod class, Mod.cc
/// \details 
/// \ref pp. 162. Edward Scheinerman.  C++ for Mathematicians: An Introduction 
///   for Students and Professionals. 2006.  Ch. 9 Modular Arithmetic.
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
#include "Mod.h"
#include "gcdx.h"

#include <iostream>

long Mod::default_modulus = INITIAL_DEFAULT_MODULUS;

std::ostream& operator<<(std::ostream& os, const Mod& M)
{
  if (!M.is_invalid())
  {
    os << "Mod(" << M.get_val() << "," << M.get_mod() << ")";
  }
  else
  {
    os << "INVALID";
  }
  return os;
}

Mod Mod::add(Mod that) const
{
  if (is_invalid() || that.is_invalid())
  {
    return Mod(0, 0);
  }

  if (mod != that.mod)
  {
    return Mod(0, 0);
  }
  return Mod(val + that.val, mod);
}

Mod Mod::multiply(Mod that) const
{
  if (is_invalid() || that.is_invalid())
  {
    return Mod(0, 0);
  }

  if (mod != that.mod)
  {
    return Mod(0, 0);
  }

  return Mod(val * that.val, mod);
}

Mod Mod::inverse() const
{
  long d, a, b;
  if (is_invalid())
  {
    return Mod(0, 0);
  }

  d = gcd(val, mod, a, b);

  if (d > 1)
  {
    return Mod(0, 0); // no reciprocal if gcd(v, x) != 1
  }
  return Mod(a, mod);
}

Mod Mod::pow(long k) const
{
  if (is_invalid())
  {
    return Mod(0, 0); // invalid is forever
  }

  // negative exponent: reciprocal and try again
  if (k < 0)
  {
    return (inverse()).pow(-k);
  }

  // zero exponent: return 1
  if (k == 0)
  {
    return Mod(1, mod);
  }

  // exponent equal to 1: return self
  if (k == 1)
  {
    return *this;
  }

  // even exponent: return (m^(k/2))^2
  if (k % 2 == 0)
  {
    Mod tmp = pow(k/2);
    return tmp * tmp;
  }

  // odd exponent: return (m^((k-1)/2))^2 * m
  Mod tmp = pow((k - 1)/2);
  return tmp * tmp * (*this);
}

