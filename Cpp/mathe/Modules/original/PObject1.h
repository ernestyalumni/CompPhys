/**
 * @file   : PObject1.h
 * @author : Ernest Yeung
 * @email  : ernestyalumni@gmail.com
 * @brief  : Program 10.4: Header file for the PObject class (version 1)
 * @ref    : pp. 186 Sec. 10.5 Class and file organization for PPoint and PLine Ch. 10 The Projective Plane; Edward Scheinerman, C++ for Mathematicians: An Introduction for Students and Professionals. Taylor & Francis Group, 2006. 
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
#ifndef POBJECT_H
#define POBJECT_H

#include <iostream>

class PObject
{
  public:
    PObject();

    PObject(const double a, const double b, const double c);

    void randomize();

    /*************************************************************************/
    /** Accessors */
    /*************************************************************************/
    const double getX() const
    {
      return x_;
    }

    const double getY() const
    {
      return y_;
    }

    const double getZ() const
    {
      return z_;
    }

    bool is_invalid() const
    {
      return (x == 0.) && (y == 0.) && (z == 0.);  
    }

  protected:
    bool equals(const PObject& that) const;
    bool less(const PObject& that) const;
    bool incident(const PObject& that) const;
    PObject rand_perp() const;
    PObject op(const PObject& that) const;

  private:
    double x_, y_, z_;
    void scale();
    double dot(const PObject& that) const;
};

std::ostream& operator<<(std::ostream& os, const PObject& A);

bool dependent(const PObject& A, const PObject& B, const PObject& C);

#endif // END of POBJECT_H