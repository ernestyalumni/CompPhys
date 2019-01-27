//------------------------------------------------------------------------------
/// \file RandomNumberGenerators.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Random number generators (RNG).
/// \url https://github.com/CompPhysics/ComputationalPhysics/blob/master/doc/Programs/LecturePrograms/programs/MCIntro/cpp/lib.cpp
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
///  g++ -std=c++17 Tuple2_main.cpp -o Tuple2_main
//------------------------------------------------------------------------------
#ifndef _MONTE_CARLO_RANDOM_NUMBER_GENERATORS_H_
#define _MONTE_CARLO_RANDOM_NUMBER_GENERATORS_H_

namespace MonteCarlo
{

namespace RandomNumberGenerators
{

namespace Details
{

constexpr long IA = 16807;
constexpr long IM = 2147483647;

} // namespace Details

//------------------------------------------------------------------------------
/// \class MinimalRandomNumber
/// \brief "Minimal" random number generator of Park and Miller
/// \ref (see Numerical recipe page 279)
/// \details Set or reset the input value idum to any integer value (except the
/// unlikely value MASK) to initialize the sequence; idum must not be altered
/// between calls for successive deviates in a sequence.
/// The function returns a uniform deviate between 0.0 and 1.0.
//------------------------------------------------------------------------------
class MinimalRandomNumber
{
  public:

    MinimalRandomNumber();

  private:


}; // class MinimalRandomNumber


} // namespace RandomNumberGenerators

} // namespace MonteCarlo

#endif // _MONTE_CARLO_RANDOM_NUMBER_GENERATORS_H_
