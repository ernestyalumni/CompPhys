//------------------------------------------------------------------------------
/// \file   : sieve.h
/// \author : Ernest Yeung
/// \email  : ernestyalumni@gmail.com
/// \brief  : The header file for a procedure to build a table of primes via 
/// the Sieve of Eratosthenes. 
/// \ref    : pp. 79 Program 5.6 Ch. 5 Arrays; Edward Scheinerman, C++ for 
///   Mathematicians: An Introduction for Students and Professionals. Taylor &
///   Francis Group, 2006. 
///
/// \copyright If you find this code useful, feel free to donate directly and
///   easily at this direct PayPal link: 
///
/// https://www.paypal.com/cgi-bin/webscr?cmd=_donations&business=ernestsaveschristmas%2bpaypal%40gmail%2ecom&lc=US&item_name=ernestyalumni&currency_code=USD&bn=PP%2dDonationsBF%3abtn_donateCC_LG%2egif%3aNonHosted 
///
/// which won't go through a 3rd. party such as indiegogo, kickstarter,
/// patreon.  
/// Otherwise, I receive emails and messages on how all my (free) material on 
/// physics, math, and engineering have helped students with their studies, and
/// I know what it's like to not have money as a student, but love physics  (or
/// math, sciences, etc.), so I am committed to keeping all my material 
/// open-source and free, whether or not sufficiently crowdfunded, under the 
/// open-source MIT license: feel free to copy, edit, paste, make your own 
/// versions, share, use as you wish.    
/// Peace out, never give up! -EY
//------------------------------------------------------------------------------
/// COMPILATION TIPS:
///   g++ -std=c++14 sieve_main.cpp sieve.cpp -o sieve_main
//------------------------------------------------------------------------------
#ifndef _SIEVE_H_
#define _SIEVE_H_

#include <cstddef> // std::size_t
#include <type_traits> // std::enable_if
#include <vector>

namespace Fields
{

//------------------------------------------------------------------------------
/// The Sieve of Eratosthenes: Generate a table of primes.
///
/// \param n upper limit on the primes (i.e., we find all primes
///   less than or equal to n).
/// \param primes array to hold the table of primes.
/// \param integer type Z, Z for Zahlen (number).
/// \ref https://en.wikibooks.org/wiki/More_C%2B%2B_Idioms/enable-if
/// This is a reference to using SFINAE (substitution-failure-is-not-an-error),
/// and function template specialization without concepts.
/// \return the number of primes we found.
//------------------------------------------------------------------------------
template<typename Z, typename = void>
Z sieve(std::size_t n, std::vector<Z>& primes);

template<
  typename Z,
  typename std::enable_if_t<std::is_integral<Z>::value>
  >
Z sieve(std::size_t n, std::vector<Z>& primes)
{
  if (n < 2)
  {
    return 0; // no primes unless n is at least 2.
  }

  // Names of marks to put in theSieve
  constexpr char blank  {0};
  constexpr char marked {1};

  // Make sure theSieve is blank to begin
  std::vector<char> theSieve(n + 1, blank);

  Z idx {0}; // index into the primes array

  for (std::size_t k {2}; k <= n; k++)
  {
    if (theSieve[k] == blank) // we found an unmarked entry
    {
      theSieve[k] = marked;     // mark it as a prime
      primes[idx] = k;      // record k in the primes array
      idx++;
    }
    // Now mark off all multiples of k 
    for (std::size_t d {2*k}; d <= n; d += k)
    {
      theSieve[d] = marked;
    }
  } // END of for (long k = 2; k <= n; k++ ) loop
  return idx;
};

template<>
long sieve(std::size_t n, std::vector<long>& primes);


} // namespace Fields

#endif // _SIEVE_H_
