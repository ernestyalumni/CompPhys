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
///  g++ -std=c++17 RandomNumberGenerators.cpp RandomNumberGenerators_main.cpp \
///   -o RandomNumberGenerators_main
//------------------------------------------------------------------------------
#ifndef _MONTE_CARLO_RANDOM_NUMBER_GENERATORS_H_
#define _MONTE_CARLO_RANDOM_NUMBER_GENERATORS_H_

#include <memory>
#include <type_traits> // std::add_lvalue_reference

namespace MonteCarlo
{

namespace RandomNumberGenerators
{

//------------------------------------------------------------------------------
/// \class RandomNumberGenerator
/// \details Follow interface inheritance hierarchy.
//------------------------------------------------------------------------------
template <typename T>
class RandomNumberGenerator
{
  public:

    virtual T operator()() = 0;
};

namespace Details
{

constexpr long IA {16807};
constexpr long IM {2147483647};
constexpr long IQ {127773};
constexpr long IR {2836};

constexpr long MASK {123459876};

constexpr double AM {1.0 / static_cast<double>(IM)};

/// For "Minimal" random number generator of Park and Miller with Bays-Durham
/// shuffle.

constexpr long NTAB {32};
constexpr long NDIV {1 + (IM - 1) / NTAB};
constexpr double EPS {1.2e-7};
constexpr double RNMX {1.0 - EPS};

/// For long period (> 2 x 10^18) random number generator of L'Ecuyer and
/// Bays-Durham shuffle and added safeguards.

constexpr long IM1 {2147483563};
constexpr long IM2 {2147483399};
constexpr long IMM1 {IM1 - 1};
constexpr long IA1 {40014};
constexpr long IA2 {40692};
constexpr long IQ1 {53668};
constexpr long IQ2 {52774};
constexpr long IR1 {12211};
constexpr long IR2 {3791};

constexpr long NDIV1 {1 + IMM1 / NTAB};

constexpr double AM1 {1.0 / static_cast<double>(IM1)};

// For uniform random number deviate between 0.0 and 1.0

constexpr long MBIG {1000000000};
constexpr long MSEED {161803398};
constexpr long MZ {0};

constexpr double FAC {1.0 / static_cast<double>(MBIG)};

} // namespace Details

//------------------------------------------------------------------------------
/// \class MinimalParkMiller
/// \brief "Minimal" random number generator of Park and Miller
/// \ref (see Numerical recipe page 279)
/// \details Set or reset the input value idum to any integer value (except the
/// unlikely value MASK) to initialize the sequence; idum must not be altered
/// between calls for successive deviates in a sequence.
/// The function returns a uniform deviate between 0.0 and 1.0.
//------------------------------------------------------------------------------
class MinimalParkMiller : public RandomNumberGenerator<double>
{
  public:

    MinimalParkMiller();

    explicit MinimalParkMiller(const long seed);

    double operator()();

    //double operator()(const long seed);

    // Accessors

    //--------------------------------------------------------------------------
    /// \details operator* provides access to the object owned by this*
    /// \url https://en.cppreference.com/w/cpp/memory/unique_ptr/operator*
    //--------------------------------------------------------------------------
    std::add_lvalue_reference<long>::type seed_value() const
    {
      return *seed_;
    }

  protected:

    //--------------------------------------------------------------------------
    /// \details Returns a pointer to the managed object, or nullptr if no
    /// object is owned.
    /// \url https://en.cppreference.com/w/cpp/memory/unique_ptr/get
    //--------------------------------------------------------------------------
    long* seed()
    {
      return seed_.get();
    }

    static constexpr long IA_ {Details::IA};
    static constexpr long IM_ {Details::IM};
    static constexpr long IQ_ {Details::IQ};
    static constexpr long IR_ {Details::IR};
    static constexpr long MASK_ {Details::MASK};
    static constexpr double AM_ {Details::AM};

  private:

    std::unique_ptr<long> seed_;
}; // class MinimalParkMiller

//------------------------------------------------------------------------------
/// \class BaysDurhamShuffle
/// \ref (see Numerical recipe pp. 280)
/// \details "Minimal" random number generator of Park and Miller with
/// Bays-Durham shuffle and added safeguards. Call with idum a negative integer
/// to initialize; thereafter, do not alter idum between successive deviates in
/// a sequence. RNMX should approximate the largest floating point value that is
/// less than 1.
/// \return The function returns a uniform deviate between 0.0 and 1.0
/// (exclusive of end-point values).
//------------------------------------------------------------------------------
class BaysDurhamShuffle : protected MinimalParkMiller
{
  public:

    BaysDurhamShuffle();

    explicit BaysDurhamShuffle(const long seed);

    double operator()();
//    double operator()(long*);

  protected:

    static constexpr long NTAB_ {Details::NTAB};
    static constexpr long NDIV_ {Details::NDIV};

    static constexpr double RNMX_ {Details::RNMX};
};

//------------------------------------------------------------------------------
/// \class LEcuyer
/// \details Long period (> 2 x 10^18) random number generator of L'Ecuyer and
/// Bays-Durham shuffle and added safeguards.
/// Call with idum a negative integer to initialize; thereafter, do not alter
/// idum between successive deviates in a sequence.
/// RNMX should approximate the largest floating pointer value that is less than
/// 1.
/// \return The function returns a uniform deviate between 0.0 and 1.0
/// (exclusive of end-point values).
//------------------------------------------------------------------------------
class LEcuyer: protected BaysDurhamShuffle
{
  public:

    LEcuyer();

    explicit LEcuyer(const long seed);

    double operator()();

  protected:

    static constexpr long IM1_ {Details::IM1};
    static constexpr long IM2_ {Details::IM2};
    static constexpr long IMM1_ {Details::IMM1};
    static constexpr long IA1_ {Details::IA1};
    static constexpr long IA2_ {Details::IA2};
    static constexpr long IQ1_ {Details::IQ1};
    static constexpr long IQ2_ {Details::IQ2};
    static constexpr long IR1_ {Details::IR1};
    static constexpr long IR2_ {Details::IR2};
    static constexpr long NDIV1_ {Details::NDIV1};

    static constexpr double AM1_ {Details::AM1};
};

//------------------------------------------------------------------------------
/// \class Uniform
/// \return Returns a uniform random number deviate between 0.0 and 1.0.
/// \details Set the idum to any negative value to initialize or reinitialize
/// the sequence. Any large MBIG, and any small (but still large) MSEED can be
/// substituted for the present values.
//------------------------------------------------------------------------------
class Uniform : public RandomNumberGenerator<double>
{
  public:

    Uniform();

    explicit Uniform(const long seed);

    double operator()();

    //--------------------------------------------------------------------------
    /// \details operator* provides access to the object owned by this*
    /// \url https://en.cppreference.com/w/cpp/memory/unique_ptr/operator*
    //--------------------------------------------------------------------------
    std::add_lvalue_reference<long>::type seed_value() const
    {
      return *seed_;
    }

  private:

    std::unique_ptr<long> seed_;

    static constexpr long MBIG_ {Details::MBIG};
    static constexpr long MSEED_ {Details::MSEED};
    static constexpr long MZ_ {Details::MZ};
    static constexpr double FAC_ {Details::FAC};
};

} // namespace RandomNumberGenerators

} // namespace MonteCarlo

#endif // _MONTE_CARLO_RANDOM_NUMBER_GENERATORS_H_
