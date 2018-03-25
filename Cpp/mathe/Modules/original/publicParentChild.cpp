/**
 * @file   : publicParentChild.cpp
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
#include <iostream>

class Parent {
  public:
    Parent(const double a, const double b):
      x_ {a}, y_ {b}
    {}

    const double sum() const
    {
      return x_ + y_;
    }

    void print() const
    {
      std::cout << "(" << x_ << "," << y_ << ")";
    }
  private:
    double x_, y_;
};

class Child : public Parent
{
  public:
    Child(const double a, const double b, const int n): 
      Parent(a, b), k_{n}
    {}
    
    const double value() const
    {
      return sum() * k_;
    }

    void print() const
    {
      Parent::print();
      std::cout << "*" << k_;
    }

  private:
    int k_;
};

int main()
{
  Parent P(3., -2.);
  P.print();
  std::cout << " --> " << P.sum() << '\n';

  Child C(-1., 3., 5);
  C.print();
  std::cout << " --> " << C.sum() << " --> " << C.value() << std::endl;

  return 0;
}