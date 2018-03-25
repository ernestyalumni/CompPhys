/**
 * @file   : Point.cpp
 * @author : Ernest Yeung
 * @email  : ernestyalumni@gmail.com
 * @brief  : A program to check the Point class. 
 * @ref    : pp. 110 Program 6.3 Ch. 6 Points in the Plane; Edward Scheinerman, C++ for Mathematicians: An Introduction for Students and Professionals. Taylor & Francis Group, 2006. 
 * @details 
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
#include "Point.h"
#include <iostream>

/**
 * @brief A main to test the Point class.
 * */

int main()
{
  Point X;          // Test constructor version 1
  Point Y(3, 4);    // Test constructor version 2

  std::cout << "The point X is " << X << " and the point Y is "
    << Y << '\n';
  std::cout << "Point Y in polar coordinates is ("
    << Y.getR() << "," << Y.getA() << ")" << '\n';

  std::cout << "The distance between these points is "
    << dist(X,Y) << '\n';
  std::cout << "The midpoint between these points is " 
    << midpoint(X, Y) << '\n';

  Y.rotate(M_PI/2);
  std::cout << "After 90-degree rotation, Y = " << Y << '\n';

  Y.setR(100);
  std::cout << "After rescaling, Y = " << Y << '\n';

  Y.setA(M_PI/4);
  std::cout << "After setting Y's angle to 45 degrees, Y = " << Y << '\n';

  Point Z;
  Z = Y; // Assign one point to another 
  std::cout << "AFter setting Z = Y, we find Z = " << Z << '\n';

  X = Point(5, 3);
  Y = Point(5, -3);

  std::cout << "Now point X is " << X << " and point Y is " << Y << '\n';
  if (X == Y)
  {
    std::cout << "They are equal." << '\n';
  }

  if (X != Y)
  {
    std::cout << "They are not equal." << std::endl;
  }

  return 0;
}