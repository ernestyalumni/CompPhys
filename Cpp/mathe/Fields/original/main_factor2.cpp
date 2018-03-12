/**
 * @file   : main_factor2.cpp
 * @author : Ernest Yeung
 * @email  : ernestyalumni@gmail.com
 * @brief  : A main to test the factor procedure. 
 * @ref    : pp. 75 Program 5.3, Ch. 5 Arrays; Edward Scheinerman, C++ for Mathematicians: An Introduction for Students and Professionals. Taylor & Francis Group, 2006. 
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
 */
#include "factor.h"
#include <iostream>

using Fields::factor;

/**
 * A program to test the factor procedure.
 * */

int main()
{

  long flist[100]; 		// place to hold the factors
  
  for (long n = 1; n <= 100; n++) 
  {
	long nfactors = factor(n, flist);
	std::cout << n << "\t";
	for (int k = 0; k < nfactors; k++)
	{
	  std::cout << flist[k] << " ";
	}
	std::cout << std::endl;
  } // END of for (long n =1; n <= 100; n++)

  int flist2[50]; 		// place to hold the factors
  
  for (int n = 1; n <= 50; n++) 
  {
	int nfactors = factor(n, flist2);
	std::cout << n << "\t";
	for (int k = 0; k < nfactors; k++)
	{
	  std::cout << flist2[k] << " ";
	}
	std::cout << std::endl;
  } // END of for (long n =1; n <= 100; n++)
}
