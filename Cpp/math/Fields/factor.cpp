/**
 * @file   : factor.cpp
 * @author : Ernest Yeung
 * @email  : ernestyalumni@gmail.com
 * @brief  : Source file factor.cc for factor procedure
 * @ref    : pp. 74 Program 5.2, Ch. 5 Arrays; Edward Scheinerman, C++ for Mathematicians: An Introduction for Students and Professionals. Taylor & Francis Group, 2006. 
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

namespace Fields
{

long factor(long n, long* flist)
{
  // If n is zero, we return -1
  if (n == 0)
  {
    return -1;
  }	
  
  // If n is negative, we change it to |n|
  if (n < 0)
  {
	n = -n;  
  }
  
  // If n is one, we simply return 0
  if (n == 1)
  {
	return 0;  
  }
  
  // At this point, we know n > 1
  int idx = 0; 		// index into the flist array
  int d = 2;		// current divisor 
  
  while (n > 1) 
  {
    while (n % d == 0) 
    {
	  flist[idx] = d;
	  ++idx;
	  n /= d;
	}
	++d;
  }
  return idx;
} // END of long factor(long, long*)

constexpr int factor_recursion(int n, int d, int idx, int* flist)
{
  if (n <= 1)
  {
	return idx;  
  }
  
  while (n % d == 0)
  {
	flist[idx] = d;
	++idx;
	n /= d;
  }
  ++d;
  
  factor_recursion(n, d, idx, flist);	
}

int factor(int n, int* flist)
{
  // If n is 0, we return -1
  if (n == 0)
  {
	return -1;
  }
  
  // If n is negative, we change it to |n|
  if (n < 0)
  {
	n = -n;
  }
  
  // If n is 1, we simply return 0
  if (n == 1)
  {
	return 0;
  }
  
  // At this point, we know n > 1
  return factor_recursion(n, 2, 0, flist);		
}

} // namespace Fields	
