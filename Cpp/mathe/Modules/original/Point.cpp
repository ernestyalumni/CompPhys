/**
 * @file   : Point.cpp
 * @author : Ernest Yeung
 * @email  : ernestyalumni@gmail.com
 * @brief  : Code for the Point class methods and procedures. 
 * @ref    : pp. 95 Program 6.1 Ch. 6 Points in the Plane; Edward Scheinerman, C++ for Mathematicians: An Introduction for Students and Professionals. Taylor & Francis Group, 2006. 
 * @details http://www.cplusplus.com/forum/beginner/105176/
 * M_PI is non-standard.
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
#include <cmath>
// #define _USE_MATH_DEFINES // M_PI is non-standard

Point::Point()
{
  x = y = 0.;
}

Point::Point(const double xx, const double yy)
{
  x = xx;
  y = yy;
}

double Point::getX() const
{
  return x;
}

double Point::getY() const
{
  return y;
}

void Point::setX(const double xx)
{
  x = xx;
}

void Point::setY(const double yy)
{
  y = yy;
}

double Point::getR() const
{
  return std::sqrt(x*x + y*y);
}

void Point::setR(const double r)
{
  // If this point is at the origin, set location to (r, 0)
  if ((x == 0.) && (y == 0.))
  {
    x = r;
    return;
  }  

  // Otherwise, set position as (r cos A, r sin A)
  double A = getA();
  x = r * std::cos(A);
  y = r * std::sin(A);
}

double Point::getA() const
{
  if ((x == 0.) && (y == 0.))
  {
    return 0.;
  }

  double A = std::atan2(y,x);
  if (A < 0)
  {
    A += 2 * M_PI;
  }
  return A;
}


void Point::setA(const double theta)
{
  double r = getR();
  x = r * std::cos(theta);
  y = r * std::sin(theta);
}

void Point::rotate(const double theta)
{
  double A = getA();
  A += theta;
  setA(A);
}

bool Point::operator==(const Point& Q) const
{
  return ((x==Q.x) && (y==Q.y));
}

bool Point::operator!=(const Point& Q) const
{
  return ! ((*this) == Q);
}

double dist(const Point P, const Point Q)
{
  double dx = P.getX() - Q.getX();
  double dy = P.getY() - Q.getY();
  return std::sqrt(dx * dx + dy * dy);
}

Point midpoint(const Point P, const Point Q)
{
  double xx = (P.getX() + Q.getX()) / 2;
  double yy = (P.getY() + Q.getY()) / 2;
  return Point(xx, yy);
}

std::ostream& operator<<(std::ostream& os, const Point& P)
{
  os << "(" << P.getX() << "," << P.getY() << ")";
  return os;
}
