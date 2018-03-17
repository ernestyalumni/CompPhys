/**
 * @file   : sieve.h
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
#ifndef SIEVE_H
#define SIEVE_H

/**
 * The Sieve of Eratosthenes: Generate a table of primes.
 * 
 * @param n upper limit on the primes (i.e., we find all primes
 * less than or equal to n).
 * @param primes array to hold the table of primes.
 * @return the number of primes we found.
 * */
long sieve(long n, long* primes);

#endif 