//------------------------------------------------------------------------------
/// \file GroupAxioms.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Group axioms closure, identity, and inverse elements.
/// \ref Ch. 21 Class Hierarchies, 21.2.Design of Class Hierarchies
///   The C++ Programming Language, 4th Ed., Stroustrup;
/// \details group (group operations, group axioms), following interface
/// implementation.
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
///  g++ -std=c++17 Tuple2_main.cpp -o Tuple2_main
//------------------------------------------------------------------------------
#ifndef _GROUPS_GROUP_AXIOMS_H_
#define _GROUPS_GROUP_AXIOMS_H_

namespace Groups
{

//------------------------------------------------------------------------------
/// \class GroupAxioms
/// \ref https://en.wikipedia.org/wiki/Group_(mathematics)
//------------------------------------------------------------------------------
template <class Element>
class GroupAxioms
{
  public:

    // Data is gone; ctors gone since there's no data to initialize.

    virtual Element group_law(const Element& b) = 0; // pure virtual function

    virtual Element identity() = 0; // pure virtual function

    virtual Element inverse(const Element& a) = 0;

    // Add virtual destructor to ensure proper cleanup of data that'll be
    // defined in derived class
    virtual ~GroupAxioms();
};

} // namespace Groups

#endif // _GROUPS_GROUP_AXIOMS_H_
