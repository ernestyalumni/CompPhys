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

Point::Point()
{
  x_ = y_ = 0.;
}

Point::Point(const double xx, const double yy) :
  x_{xx}, y_{yy}
{}

double Point::getX() const
{
  return x_;
}

double Point::getY() const
{
  return y_;
}

void Point::setX(const double x)
{
  x_ = x;
}

void Point::setY(const double y)
{
  y_ = y;
}

double Point::getR() const
{
  return std::sqrt(x_ * x_ + y_ * y_);
}

void Point::setR(const double r)
{
  // If this point is at the origin, set location to (r, 0)
  if ((x_ == 0.) && (y_ == 0.))
  {
    x_ = r;
    return;
  }  

  // Otherwise, set position as (r cos A, r sin A)
  double A {getA()};
  x_ = r * std::cos(A);
  y_ = r * std::sin(A);
}

double Point::getA() const
{
  if ((x_ == 0.) && (y_ == 0.))
  {
    return 0.;
  }

  double A {std::atan2(y_, x_)};
  if (A < 0)
  {
    A += 2 * M_PI;
  }
  return A;
}

void Point::setA(const double theta)
{
  double r {getR()};
  x_ = r * std::cos(theta);
  y_ = r * std::sin(theta);
}

void Point::rotate(const double theta)
{
  double A {getA()};
  A += theta;
  setA(A);
}

Point& Point::translate(const double dx, const double dy)
{
  x_ += dx;
  y_ += dy;
  return (*this);
}

bool Point::operator==(const Point& Q) const
{
  return ((x_ == Q.x_) && (y_ == Q.y_));
}

bool Point::operator!=(const Point& Q) const
{
  return !((*this) == Q);
}

double dist(const Point P, const Point Q)
{
  double dx {P.getX() - Q.getX()};
  double dy {P.getY() - Q.getY()};
  return std::sqrt(dx * dx + dy * dy);
}

Point midpoint(const Point P, const Point Q)
{
  double x {(P.getX() + Q.getX()) / 2.};
  double y {(P.getY() + Q.getY()) / 2.};
  return Point(x, y);
}

std::ostream& operator<<(std::ostream& os, const Point& P)
{
  os << "(" << P.getX() << "," << P.getY() << ")";
  return os;
}
