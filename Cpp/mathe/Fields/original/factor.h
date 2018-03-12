/**
 * @file   : factor.h
 * @author : Ernest Yeung
 * @email  : ernestyalumni@gmail.com
 * @brief  : Header file for factor procedure
 * @ref    : pp. 73 Program 5.1 Ch. 5 Arrays; Edward Scheinerman, C++ for Mathematicians: An Introduction for Students and Professionals. Taylor & Francis Group, 2006. 
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
#ifndef _FIELDS_FACTOR_H_
#define _FIELDS_FACTOR_H_

namespace Fields
{

/**
 * Factor an integer n. The prime factors are saved in the second 
 * argument, flist. It is the user's responsibility to be sure that 
 * flist is large enough to hold all the primes. If n is negative, we 
 * factor -n instead. If n is zero, we return -1. The case n equal to 
 * 1 causes this procedure to return 0 and no primes are saved in 
 * flist.
 * 
 * @param n the integer we wish to factor
 * @param flist an array to hold the prime factors
 * @return the number of prime factors
 */
long factor(long n, long* flist);

// overloaded version
int factor(int n, int* flist);

} // namespace Fields

#endif // FIELDS_FACTOR_H__
