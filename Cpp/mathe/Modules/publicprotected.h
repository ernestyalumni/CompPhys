/**
 * @file   : publicprotected.h
 * @author : Ernest Yeung
 * @email  : ernestyalumni@gmail.com
 * @brief  : Program 10.1: A program to illustrate inheritance
 * @ref    : pp. 181 Sec. 10.3 Inheritance Ch. 10 The Projective Plane; Edward Scheinerman, C++ for Mathematicians: An Introduction for Students and Professionals. Taylor & Francis Group, 2006. 
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
#include <iostream>

/*****************************************************************************/
/** public inheritance */
/*****************************************************************************/

class Parent 
{
  public:
    explicit Parent(const double a, const double b);

    const double sum() const
    {
      return x_ + y_;
    }

    void print() const
    {
      std::cout << "(" << x_ << "," << y_ << ")";
    }

    /*************************************************************************/
    /** Accessors */
    /*************************************************************************/
    const double get_x_() const
    {
      return x_;
    }

  private:
    double x_;
    double y_ {};
}; // END of class Parent

class Child_public : public Parent
{
  public:
    
    using Parent::get_x_;
  
    explicit Child_public(const double a, const double b, const int n);

    const double value() const
    {
      return sum() * k_;
    }

    void print() const;

  private:
    int k_;
}; // END of class Child_public

/*****************************************************************************/
/** "protected" inheritance */
/*****************************************************************************/
class Base
{
  public:
    explicit Base(const int x, const int y);

    void print() const
    {
      std::cout << "(" << a_ << "," << b_ << ")";
    }  

    /*************************************************************************/
    /** Accessors */
    /*************************************************************************/
    const int get_a_() const
    {
      return a_;      
    }    

    const int get_b_() const
    {
      return b_;      
    }    

  protected:
    int sum() const
    {
      return a_ + b_;
    }

    /*************************************************************************/
    /** Accessors */
    /*************************************************************************/
    const int get_a_protected() const
    {
      return a_;      
    }    

    const int get_b_protected() const
    {
      return b_;      
    }    

    int b_;

  private:
    int a_;
}; // END of class Base

class Child: public Base
{
  public:
    
    using Base::get_a_;
    using Base::get_b_;
    using Base::get_a_protected; 
    using Base::get_b_protected;
    
     explicit Child(const int x, const int y);

    void increase_b()
    {
      b_++;
    }

    void print() const;    
};

class GrandChild : Child
{
  public:

    using Child::get_a_;
    using Base::get_b_;
    using Child::get_a_protected; 
    using Base::get_b_protected;

    explicit GrandChild(const int x, const int y, const int z);

    void print() const;

  private:
    int c_;
};