/**
 * @file   : sieve.cpp
 * @author : Ernest Yeung
 * @email  : ernestyalumni@gmail.com
 * @brief  : The header file for a procedure to build a table of primes via the
 * Sieve of Eratosthenes. 
 * @ref    : pp. 79 Program 5.6 Ch. 5 Arrays; Edward Scheinerman, C++ for Mathematicians: An Introduction for Students and Professionals. Taylor & Francis Group, 2006. 
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
 * COMPILATION TIPS:
 *  g++ -std=c++17 -c factor.cpp
 * */
#include "sieve.h"

//template <typename T, long TABLE_SIZE>
//long sieve(long n, T(&primes)[TABLE_SIZE])
long sieve(const long n, long primes[])
{
  if (n < 2)
  {
    return 0;   // no primes unless n is at least 2.
  }

  char* sieve {new char[n + 1]}; // hold the marks

  // Names of marks to put in the sieve
  constexpr const char blank {0};
  constexpr const char marked {1};
 
  // Make sure sieve is blank to begin
  for (long p = 2; p <= n; p++)
  {
    sieve[p] = blank;
  }

  long idx = 0;     // index into the primes array

  for (long p = 2; p <= n; p++)
  {
    if (sieve[p] == blank)      // we found an unmarked entry
    {
      sieve[p] = marked;        // mark it as a prime
      primes[idx] = p;          // record p in the primes array
      idx++;
    }
    // Now mark off all multiples of p
    for (long q = 2*p; q <= n; q += p)
    {
      sieve[q] = marked;
    }
  } // END of for k loop
  delete[] sieve;
  return idx;
}
