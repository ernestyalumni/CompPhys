/**
 * @file   : sieve_main.cpp
 * @author : Ernest Yeung
 * @email  : ernestyalumni@gmail.com
 * @brief  : A program to test the sieve procedure.
 * @ref    : pp. 82 Program 5.8 Ch. 5 Arrays; Edward Scheinerman, C++ for Mathematicians: An Introduction for Students and Professionals. Taylor & Francis Group, 2006. 
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
#include <iostream>

const long N = 10000000;            // ten million
const long TABLE_SIZE = 800000;     // prime number theorem overestimate

/**
 * A program to test the sieve procedure.
 * */
//using sieve<long, TABLE_SIZE>;
//template <>
//long sieve<long, TABLE_SIZE>(long, long(&)[TABLE_SIZE]);

int main(int argc, char* argv[]) 
{
  long primes[TABLE_SIZE];
//  long np = sieve<long, TABLE_SIZE>(N, primes);
//  long np = sieve_long(N, primes);
  long np {sieve(N, primes)} ;

  std::cout << "We found " << np << " primes " << '\n';

  std::cout << "The first 10 primes we found are these: " << '\n';

  for (long k = 0; k < 10; k++)
  {
    std::cout << primes[k] << " ";
  }
  std::cout << '\n';

  std::cout << "The largest prime we found is " << primes[np -1] << std::endl;

  return 0;
}