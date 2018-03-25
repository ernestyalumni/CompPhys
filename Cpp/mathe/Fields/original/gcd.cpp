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
#include <iostream>

namespace Fields
{

long gcd(long a, long b)
{
    // if a and b are both zero, print an error and return 0
    if ( (a == 0) && (b == 0) ) 
    {
        std::cerr << "WARNING: gcd called with both arguments equal to zero." 
            << std::endl;
        // It would not have been a mistake to use cout here instead.
        // The cout is usually used for standard output and cerr for error messages
        return 0;
    }

    // Make sure a and b are both nonnegative
    if (a < 0)
    {
        a = -a;
    }
    if (b < 0)
    {
        b = -b;
    }
    // no change made to any values outside gcd procedure; no side effects
    // when another procedure (say, main()) calls gcd, arguments are copies to a and b

    // if a is zero, the answer is b
    if (a == 0)
    {
        return b;
    }

    // otherwise, we check all possibilities from 1 to a
    long d;     // d will hold the answer 

    for (long t = 1; t <= a; t++)
    {
        if ( (a % t == 0) && (b % t == 0))
        {
            d = t;
        }
    }

    return d;
}

} // namespace Fields