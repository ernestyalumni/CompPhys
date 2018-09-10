//------------------------------------------------------------------------------
/// \file   : Point_main.cpp
/// \author : Ernest Yeung
/// \email  : ernestyalumni@gmail.com
/// \brief  : Program 6.3: A program to check the Point class.
/// \details : A main to test the Point class. Concrete class - defining
///   property is its representation is its definition
/// \ref    : 3.2.1.1 An Arithmetic Type, Ch. 3 A Tour of C++: Abstraction 
///   Mechanisms. Bjarne Stroustrup, The C++ Programming Language, 4th Ed.
///   https://stackoverflow.com/questions/8752837/undefined-reference-to-template-class-constructor
/// \copyright If you find this code useful, feel free to donate directly and
/// easily at this direct PayPal link: 
///
/// https://www.paypal.com/cgi-bin/webscr?cmd=_donations&business=ernestsaveschristmas%2bpaypal%40gmail%2ecom&lc=US&item_name=ernestyalumni&currency_code=USD&bn=PP%2dDonationsBF%3abtn_donateCC_LG%2egif%3aNonHosted 
/// 
/// which won't go through a 3rd. party such as indiegogo, kickstarter, patreon.  
/// Otherwise, I receive emails and messages on how all my (free) material on
/// physics, math, and engineering have helped students with their studies, and
/// I know what it's like to not have money as a student, but love physics (or 
/// math, sciences, etc.), so I am committed to keeping all my material 
/// open-source and free, whether or not sufficiently crowdfunded, under the 
/// open-source MIT license: feel free to copy, edit, paste, make your own
/// versions, share, use as you wish.    
/// Peace out, never give up! -EY
//------------------------------------------------------------------------------
/// COMPILATION TIPS:
///   g++ -std=c++14 Point.cpp Point_main.cpp -o Point_main
//------------------------------------------------------------------------------
#include "Point.h"

#include <cmath>
#include <iostream>

using AffineSpace::Point;

class TestPoint : public Point
{
  public:
    using Point::Point;
    using Point::set_R;
    using Point::set_theta;
};

int main()
{

  //----------------------------------------------------------------------------
  /// PointDefaultConstructs
  Point p;

  // Test constructor version 1, default constructor
  Point X; 

  // Test constructor version 2
  TestPoint Y {3, 4}; 

  std::cout << "The point X is " << X << " and the point Y is "
    << Y << std::endl;

  std::cout << "Point Y in polar coordinates is ("
    << Y.get_R() << "," << Y.get_theta() << ")" << std::endl;

  std::cout << "The distance between these points is " << dist(X, Y) << '\n';

  std::cout << "The midpoint between these points is " << midpoint(X, Y) << 
    '\n';
  
  Y.rotate(M_PI / 2);
  std::cout << "After 90-degree rotation, Y = " << Y << '\n';

  Y.set_R(100);
  std::cout << "After rescaling, Y = " << Y << '\n';

  Y.set_theta(M_PI / 4);
  std::cout << "After setting Y's angle to 45 degrees, Y = " << Y << '\n';

  Point Z;
  Z = Y; // Assign one point to another.
  std::cout << "After setting Z = Y, we find Z = " << Z << '\n';

  X = Point(5, 3);
  Y = TestPoint(5, -3);

  std::cout << "Now point X is " << X << " and point Y is " << Y << '\n';
  if (X == Y)
  {
    std::cout << "They are equal." << std::endl;
  }

  if (X != Y)
  {
    std::cout << "They are not equal." << std::endl;
  }

  return 0;
}
