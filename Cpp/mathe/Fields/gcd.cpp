/**
 * @file   : gcd.cpp
 * @author : Ernest Yeung
 * @email  : ernestyalumni@gmail.com
 * @brief  : GCD of 2 integers
 * @ref    : pp. 159 Ch. 9 Modular Arithmetic; Edward Scheinerman, C++ for Mathematicians: An Introduction for Students and Professionals. Taylor & Francis Group, 2006. 
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
 * 	feel free to copy, edit, paste, make your own versions, share, use as you wish.    
 * Peace out, never give up! -EY
 * 
 * */
#include "gcd.h"

namespace Fields
{

/**
 * @name gcd_recursion
 * @brief recursion step for gcd
 * */
constexpr long gcd_recursion(const long a, const long b)
{
  return (b == 0) ? a : gcd_recursion(b, a % b);
}

// overloaded version
constexpr int gcd_recursion(const int a, const int b)
{
  return (b == 0) ? a : gcd_recursion(b, a % b);
}

long gcd(long a, long b)
{
  // Make sure a and b are both nonnegative
  if (a < 0)
  {
    a = -a;
  }
  if (b < 0)
  {
    b = -b;
  }

  // if a and b are both zero, return 0 
  if ( (a == 0) && (b == 0) )
  {
    return 0;
  }

  // If a is zero, the answer is b
  if (a == 0)
  {
    return b;
  }

  return gcd_recursion(a, b);
}

int gcd(int a, int b)
{
  // Make sure a and b are both nonnegative
  if (a < 0)
  {
    a = -a;
  }
  if (b < 0)
  {
    b = -b;
  }

  // if a and b are both zero, return 0 
  if ( (a == 0) && (b == 0) )
  {
    return 0;
  }

  // If a is zero, the answer is b
  if (a == 0)
  {
    return b;
  }

  return gcd_recursion(a, b);
}


} // namespace Fields