/**
 * @file   : protectedParentChild.cpp
 * @author : Ernest Yeung
 * @email  : ernestyalumni@gmail.com
 * @brief  : Program 10.2: A program to illustrate the use of protected members of a class
 * @ref    : pp. 184 Sec. 10.4 Protected class members Ch. 10 The Projective Plane; Edward Scheinerman, C++ for Mathematicians: An Introduction for Students and Professionals. Taylor & Francis Group, 2006. 
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

class Base
{
  public:
    Base(int x, int y):
      a_{x}, b_{y}
    {}

   void print() const
   {
     std::cout << "(" << a_ << "," << b_ << ")"; 
   }

  protected:
    int b_;

    int sum() const
    {
      return a_ + b_;
    }

  private:
    int a_;
};

class Child : public Base
{
  public:
    
    Child(int x, int y):
      Base(x, y)
    {}

  void increase_b()
  {
    b_++;
  }

  void print() const
  {
    Base::print(); 
    std::cout << "=" << sum();
  }
};

class GrandChild : public Child
{
  public:
    GrandChild(int x, int y, int z):
      Child(x,y), c_{z}
    {}

    void print() const
    {
      Base::print(); 
      std::cout << "/" << c_;
    }

  private:
    int c_;
};

int main()
{
  Base          B(1, 2);
  Child         C(3, 4);
  GrandChild    D(5, 6, 7);

  B.print(); 
  std::cout << '\n';

  C.print();
  std::cout << "  -->  ";
  C.increase_b();
  C.print(); 
  std::cout << '\n';

  D.print();
  std::cout << "  -->  ";
  D.print();
  std::cout << std::endl;

  return 0;
}