/**
 * @file   : Point.h
 * @author : Ernest Yeung
 * @email  : ernestyalumni@gmail.com
 * @brief  : The header file Point.h for the Point class (condensed version)
 * @ref    : pp. 95 Program 6.1 Ch. 6 Points in the Plane; Edward Scheinerman, C++ for Mathematicians: An Introduction for Students and Professionals. Taylor & Francis Group, 2006. 
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
#ifndef POINT_H
#define POINT_H
#include <iostream>

constexpr const double M_PI {3.14159265358979323846264338327950288419716939937510};

class Point
{
  public:
    Point(); // default constructor (ctor)
    Point(const double x, const double y); // user-defined ctor
    
    Point(const Point&) = default;
    Point(Point&&) = default;    
    Point& operator=(const Point&) = default;
    Point& operator=(Point&&) = default;
    ~Point() = default;

    //-------------------------------------------------------------------------
    /// Accessors
    //-------------------------------------------------------------------------
    double getX() const;
    double getY() const;

    void setX(const double x);
    void setY(const double y);

    //-------------------------------------------------------------------------
    /// Accessors
    //-------------------------------------------------------------------------
    double getR() const;
    double getA() const;

    void setR(const double r);
    void setA(const double theta);
    
    void rotate(const double theta);

    // move the point from its current location (x,y) to 
    // new location (x+dx,y+dy)
    Point& translate(const double dx, const double dy);

    bool operator==(const Point& Q) const;
    bool operator!=(const Point& Q) const;

  private:
    double x_;
    double y_;
};

double dist(const Point P, const Point Q);
Point midpoint(const Point P, const Point Q);
std::ostream& operator<<(std::ostream& os, const Point& P);

#endif // POINT_H