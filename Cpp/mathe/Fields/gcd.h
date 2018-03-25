/**
 * @file   : gcd.h
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
#ifndef __GCD_H__
#define __GCD_H__

namespace Fields
{

/**
 * @brief Calculate the greatest common divisor of 2 integers.
 * @details Note: gcd(0, 0) will return 0 and print an error message.
 * @param a the first integer
 * @param b the second integer
 * @return the greatest common divisor of a and b
 * */
long gcd(long a, long b);

// overload
int gcd(int a, int b);

} // namespace Fields

#endif // __GCD_H__ this line and previous are the mechanism to prevent double inclusion