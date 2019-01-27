//------------------------------------------------------------------------------
/// \file Ring.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Ring.
/// \ref Ch. 21 Class Hierarchies, 21.2.Design of Class Hierarchies
///   The C++ Programming Language, 4th Ed., Stroustrup;
/// \details Ring, following interface implementation.
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
#ifndef _RINGS_RING_H_
#define _RINGS_RING_H_

namespace Rings
{

//------------------------------------------------------------------------------
/// \class RingOperations
/// \brief The 2 binary operations every ring is equipped with
/// \details Ring binary operations (2 total) as pure interface.
/// \ref https://en.wikipedia.org/wiki/Ring_(mathematics)
//------------------------------------------------------------------------------
template <class Element>
class RingOperations
{
  public:

    // Data is gone; ctors gone since there's no data to initialize.

    //--------------------------------------------------------------------------
    /// binary operations
    /// \details pure virtual function is a virtual function whose declarator
    /// has syntax declarator virt-specifier(optional) = 0
    /// so that class immediately is an abstract class (at least has 1 abstract
    /// class).
    /// \ref https://en.cppreference.com/w/cpp/language/operator_arithmetic
    /// https://en.cppreference.com/w/cpp/language/abstract_class
    //--------------------------------------------------------------------------

    virtual Element operator+(const Element& b) = 0; // pure virtual function

    virtual Element operator*(const Element& c) = 0; // pure virtual function

    virtual Element additive_identity() const = 0; // pure virtual function

    virtual Element additive_inverse(const Element& a) = 0;

    virtual Element multiplicative_identity() const = 0;

    // Add virtual destructor to ensure proper cleanup of data that'll be
    // defined in derived class
    virtual ~RingOperations();
};

} // namespace Rings

#endif // _RINGS_RING_H_
