/**
 * @file   : sieve.cpp
 * @author : Ernest Yeung
 * @email  : ernestyalumni@gmail.com
 * @brief  : The sieve procedure
 * @ref    : pp. 81 Program 5.7 Ch. 5 Arrays; Edward Scheinerman, C++ for Mathematicians: An Introduction for Students and Professionals. Taylor & Francis Group, 2006. 
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

long sieve(long n, long* primes)
{
  if (n < 2)
  {
    return 0; // no primes unless n is at least 2.
  }

  char* theSieve;

  theSieve = new char[n+1];     // hold the marks

  // Names of marks to put in theSieve
  const char blank  = 0;
  const char marked = 1;

  // Make sure theSieve is blank to begin
  for (long k = 2; k <= n; k++)
  {
    theSieve[k] = blank;
  }

  long idx = 0; // index into the primes array

  for (long k = 2; k <= n; k++)
  {
    if (theSieve[k] == blank) // we found an unmarked entry
    {
      theSieve[k] = marked;     // mark it as a prime
      primes[idx] = k;      // record k in the primes array
      idx++;
    }
    // Now mark off all multiples of k 
    for (long d= 2*k; d <= n; d += k)
    {
      theSieve[d] = marked;
    }
  } // END of for (long k = 2; k <= n; k++ ) loop
  delete[] theSieve;
  return idx;
}