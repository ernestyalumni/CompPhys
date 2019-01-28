//------------------------------------------------------------------------------
/// \file Function.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Implementing (mathematical) functions.
/// \url
/// \ref Ch. 21 Class Hierarchies, 21.2.Design of Class Hierarchies
///   The C++ Programming Language, 4th Ed., Stroustrup;
/// \details Random number generators..
/// \copyright If you find this code useful, feel free to donate directly
/// (username ernestyalumni or email address above), going directly to:
///
/// paypal.me/ernestyalumni
///
/// which won't go through a 3rd. party like indiegogo, kickstarter, patreon.
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
///  g++ -std=c++17 RandomNumberGenerators.cpp RandomNumberGenerators_main.cpp \
///   -o RandomNumberGenerators_main
//------------------------------------------------------------------------------
#ifndef _RINGS_FUNCTION_H_
#define _RINGS_FUNCTION_H_

#include <functional>

namespace Rings
{

//------------------------------------------------------------------------------
/// \url https://en.cppreference.com/w/cpp/utility/functional/function
/// \details Class template std::function is a general-purpose polymorphic
/// function wrapper, instances of std::function can store, copy, and invoke
/// any Callable target - functions, lambda expressions, bind expression, or
/// other function objects, as well as pointers to member functions and pointers
/// to data members.
//------------------------------------------------------------------------------
template <typename Domain, typename Codomain>
class Function : public std::function<Codomain(Domain)>
{
  public:

    using std::function<Codomain(Domain)>::function;
};

} // namespace Rings

#endif // _RINGS_FUNCTION_H_
