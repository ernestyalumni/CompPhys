/**
 * @file   : gcd-tested.cpp
 * @author : Ernest Yeung
 * @email  : ernestyalumni@gmail.com
 * @brief  : GCD of 2 integers
 * @ref    : Ch. 3 Greatest Common Divisor; Edward Scheinerman, C++ for Mathematicians: An Introduction for Students and Professionals. Taylor & Francis Group, 2006. 
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
 * COMPILATION TIPS
 * 
 * */
#include "gcd.h"
#include <iostream>

//using namespace Fields;
using Fields::gcd;

/**
 * A program to test the gcd procedure
 * */
int main()
{
    long a, b;

    std::cout << "Enter the first number  --> ";
    std::cin >> a;
    std::cout << "Enter the second number --> ";
    std::cin >> b;

    std::cout << "The gcd of " << a << " and " << b << " is "
                << gcd(a,b) << std::endl;
    return 0;
}
