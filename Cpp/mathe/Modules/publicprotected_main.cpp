/**
 * @file   : publicprotected_main.cpp
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
#include "publicprotected.h"

int main()
{
  /***************************************************************************/
  /** public inheritance */
  /***************************************************************************/
  Parent P(3., -2.);
  P.print();
  std::cout << " --> " << P.sum() << '\n';

  Child_public C_public(-1., 3., 5);
  C_public.print();
  std::cout << " --> " << C_public.sum() << " --> " << C_public.value() << '\n';

  /***************************************************************************/
  /** "protected" inheritance */
  /***************************************************************************/
  
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

  std::cout << '\n' << " Test Accessors : " << '\n';
  std::cout << C.get_a_() << " " << C.get_b_() << " " << 
    C.get_a_protected() << " " << C.get_b_protected() << '\n';

  std::cout << D.get_a_() << " " << D.get_b_() << " " << 
    D.get_a_protected() << " " << D.get_b_protected() << '\n';

  return 0;
}