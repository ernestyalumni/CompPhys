//------------------------------------------------------------------------------
/// \file   : Point.cpp
/// \author : Ernest Yeung
/// \email  : ernestyalumni@gmail.com
/// \brief  : Program 6.2: Code for the Point class methods and procedures.
/// \ref    : pp. 109. Edward Scheinerman.  C++ for Mathematicians: An
///   Introduction for Students and Professionals. 2006.  Ch. 6 Points in the
///   Plane.
/// \details : Using RAII for Concrete classes. 
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
///   g++ -std=c++14 ComplexNumber_main.cpp ComplexNumber.cpp -o ComplexNumber_main
//------------------------------------------------------------------------------
#include "Point.h"

#include <cmath>
#include <iostream>

namespace AffineSpace
{

Point::Point() :
  x_{0.},
  y_{0.}
{}

Point::Point(const double x, const double y):
  x_{x},
  y_{y}
{}

const double Point::get_R() const
{
  return std::sqrt(x_ * x_ + y_ * y_);
}

void Point::set_R(const double r)
{
  // If this point is at the origin, set location to (r, 0)
  if ((x_ == 0.) && (y_ == 0.))
  {
    x_ = r;
    return;
  }

  // Need to calculate original theta before setting x, y coordinates.
  const double theta {get_theta()};
  // Otherwise, set position as (r cos theta, r sin theta)
  x_ = r * cos(theta);
  y_ = r * sin(theta);
}

const double Point::get_theta() const
{
  if ((x_ == 0.) && (y_ == 0.))
  {
    return 0.;
  }

  double theta {atan2(y_, x_)};
  if (theta < 0)
  {
    theta += 2 * M_PI;
  }
  return theta;
}

void Point::set_theta(const double theta)
{
  const double r {get_R()};
  x_ = r * cos(theta);
  y_ = r * sin(theta);
}

void Point::rotate(const double theta)
{
  set_theta(get_theta() + theta);
}

bool Point::operator==(const Point& Q) const
{
  return ((x_ == Q.x_) && (y_ == Q.y_));
}

bool Point::operator!=(const Point& Q) const
{
  return !((*this) == Q);
}

double dist(const Point& P, const Point& Q)
{
  const double dx { P.x() - Q.x()};
  const double dy { P.y() - Q.y()};
  return std::sqrt( dx * dx + dy * dy);
}

Point midpoint(const Point& P, const Point& Q)
{
  const double x {(P.x() + Q.x())/2.};
  const double y {(P.y() + Q.y())/2.};
  return Point(x, y);
}

std::ostream& operator<<(std::ostream& os, const Point& P)
{
  os << "(" << P.x() << "," << P.y() << ")";
  return os;
}

} // namespace AffineSpace
