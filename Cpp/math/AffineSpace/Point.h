//------------------------------------------------------------------------------
/// \file Point.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Program 6.1: Header file Point.h for the Point class
/// \ref pp. 95. Edward Scheinerman.  C++ for Mathematicians: An Introduction 
///   for Students and Professionals. 2006.  Ch. 6 Points in the Plane.
/// \details It can be useful to have different actions taken for lvalues and
///  rvalues. Consider a class for holding an {integer, pointer} pair. 
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
///  g++ -std=c++14 Socket_main.cpp -o Socket_main
//------------------------------------------------------------------------------
#ifndef _AFFINE_SPACE_POINT_H_
#define _AFFINE_SPACE_POINT_H_

#include <iostream>

namespace AffineSpace
{

class Point
{
  public:

    Point();
    Point(const double x, const double y);

    const double x() const
    {
      return x_;
    }    

    const double y() const
    {
      return y_;
    }

    const double get_R() const;

    const double get_theta() const;

    void rotate(const double theta);

    bool operator==(const Point& Q) const;

    bool operator!=(const Point& Q) const;

  protected:

    void set_x(const double x)
    {
      x_ = x;
    }

    void set_y(const double y)
    {
      y_ = y;
    }

    void set_R(const double r);

    void set_theta(const double theta);

  private:

    double x_;
    double y_;
};

double dist(const Point& P, const Point& Q);

Point midpoint(const Point& P, const Point& Q);

std::ostream& operator<<(std::ostream& os, const Point& P);

} // namespace AffineSpace

#endif // _AFFINE_SPACE_POINT_H_
