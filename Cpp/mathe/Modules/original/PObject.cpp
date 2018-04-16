/**
 * @file   : PObjec1t.cpp
 * @author : Ernest Yeung
 * @email  : ernestyalumni@gmail.com
 * @brief  : Program 10.5: Program file for the PObject class (version 1)
 * @ref    : pp. 192 Sec. 10.5 Class and file organization for PPoint and PLine Ch. 10 The Projective Plane; Edward Scheinerman, C++ for Mathematicians: An Introduction for Students and Professionals. Taylor & Francis Group, 2006. 
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
 * */
#include "PObject.h"

/** @ref http://en.cppreference.com/w/cpp/numeric/random/uniform_real_distribution */
#include <random>

PObject::PObject():
  x_{0.}, y_{0.}, z_{1.}
{}

PObject::PObject(const double a, const double b, const double c):
  x_{a}, y_{b}, z_{c}
{
  scale();
}

void PObject::scale()
{
  if (z_ != 0.)
  {
    x_ /= z;
    y_ /= z;
    z_ = 1.;
    return;
  }
  if (y_ != 0.) // implied that z_ == 0.
  {
    x_ /= y_;
    y_ = 1.;
    return;
  }
  if (x_ != 0.) // implied that z_ == 0, y_ == 0
  {
    x_ = 1.;
  }
}

double PObject::dot(const PObject& that) const 
{
  return x * that.x + y * that.y + z * that_z;
}

void PObject::randomize()
{
  std::random_device rd; // Will be used to obtain a seed for the random number engine 
  std::mt19937 gen(rd()); // Standard mersenne_twister engine seeded with rd()
  std::uniform_real_distribution<> dis(-1.0, 1.0);
  do 
  {
    x_ = dis(gen);
    y_ = dis(gen);
    z_ = dis(gen);
  } while (x_ * x_ + y_ * y_ + z_ * z_) > 1.);
  scale();
}

bool PObject::equals(const PObject& that) const
{
  return ((x_ == that.x_) && (y_ == that.y_ ) && (z_ == that.z_));
}

bool PObject::less(const PObject& that) const
{
  if (x_ < that.x_)
  {
    return true;
  }
  if (x_ > that.x_)
  {
    return false;
  }
  if (y_ < that.y_)
  {
    return true;
  }
  if (y_ > that.y_)
  {
    return false;
  }
  if (z_ < that.z_)
  {
    return true;
  }
  return false;
}

bool PObject::incident(const PObject& that) const
{
  return dot(that) == 0.;
}

std::ostream& operator<<(std::ostream& os, const PObject& A)
{
  os << "<"
    << A.getX() << ","
    << A.getY() << ","
    << A.getZ()
    << ">";
  return os;
}

/** @brief determinant */
bool dependent(const PObject& A, const PObject& B, const PObject& C)
{
  double a1 {A.getX()};
  double a2 {A.getY()};
  double a3 {A.getZ()};

  double b1 {B.getX()};
  double b2 {B.getY()};
  double b3 {B.getZ()};

  double c1 {C.getX()};
  double c2 {C.getY()};
  double c3 {C.getZ()};

  double det = a1*b2*c3 + a2*b3*c1 + a3*b1*c2
    - a3*b2*c1 - a1*b3*c2 - a2*b1*c3;
  return det == 0.;
}

PObject PObject::rand_perp() const
{
  if (is_invalid())
  {
    return PObject(0, 0, 0);
  }
  double x1, y1, z1;    // One vector orthogonal to (x, y, z)
  double x2, y2, z2;    // Another orthogonal to (x, y, z) and (x1, y1, z1)

  if (z_ == 0.) // If z_ == 0, take (0, 0, 1) for (x1, y1, z2)
  {
    x1 = 0;
    y1 = 0;
    z1 = 1;
  }
  else
  {
    if (y_ == 0.) // z_ != 0 and y_ == 0, use (0, 1, 0)
    {
      x1 = 0;
      y1 = 1;
      z1 = 1;
    }
    else // y and z both nonzero, use (0, -z, y)
    { 
      x1 = 0;
      y1 = -z;
      z1 = y;
    }
  }

  // normalize (x1, y1, z1)
  double r1 {std::sqrt(x1*x1 + y1*y1 + z1*z1)};
  x1 /= r1;
  y1 /= r1;
  z1 /= r1;

  // (get x2, y2, z2) by cross product with (x_, y_, z_) and (x1, y1, z1)
  x2 = -(y1*z_) + y_*z1;
  y2 = x1*z_ - x_*z1;
  z2 = -(x1*y_) + x_*y1;

  // normalize (x2, y2, z2)
  double r2 {std::sqrt(x2*x2 + y2*y2 + z2*z2)};
  x2 /= r2;
  y2 /= r2;
  z2 /= r2;

  // get a point uniformly on the unit circle
  double a, b, r;
  do {
  std::random_device rd; // Will be used to obtain a seed for the random number engine 
  std::mt19937 gen(rd()); // Standard mersenne_twister engine seeded with rd()
  std::uniform_real_distribution<> dis(-1.0, 1.0);
    a = dis(gen);
    b = dis(gen);
    r = a*a + b*b;
  } while (r > 1.);
  r = std::sqrt(r);
  a /= r;
  b /= r;

  double xx {x1 * a + x2 * b};
  double yy {y1 * a + y2 * b};
  double xx {z1 * a + z2 * b};
  
  return PObject(xx, yy, zz);
} // END of PObject::rand_perp

PObject PObject::op(const PObject& that) const
{
  if (equals(that))
  {
    return PObject(0, 0, 0);
  }

  double c1 {y*that.z - z*that.y};
  double c2 {z*that.x - x*that.z};
  double c3 {x*that.y - y*that.x};

  return PObject(c1, c2, c3);
} 

