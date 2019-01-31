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
///  g++ -std=c++17 -I ../ RandomNumberGenerators.cpp Integrate1_main.cpp \
///   -o RandomNumberGenerators_main
//------------------------------------------------------------------------------
#ifndef _MONTE_CARLO_INTEGRATE1_H_
#define _MONTE_CARLO_INTEGRATE1_H_

#include "RandomNumberGenerators/RandomNumberGenerators.h"
#include "Rings/Function.h"

#include <cassert>
#include <vector> // For debug purposes

namespace MonteCarlo
{

template <typename Domain, typename Codomain>
class Integrate
{
  public:

    using F = Rings::Function<Domain, Codomain>;

    //--------------------------------------------------------------------------
    /// \fn reset
    /// \brief Set the total number of samples to run and reset the values to
    /// sum over (2 of them, the running total and sum of sigma)
    //--------------------------------------------------------------------------
    virtual void reset(const long total_number_of_samples) = 0;

    virtual void set_f(const F&) = 0;

    virtual void run_Monte_Carlo() = 0;
}; // Integrate1

template <typename Domain, typename Codomain, class RandomNumberGenerator>
class Integrate1 : public Integrate<Domain, Codomain>
{
  public:

//    using Integrate<Domain, Codomain>::F;
    using F = Rings::Function<Domain, Codomain>;

    Integrate1() = delete;

    Integrate1(const F& f, const long total_number_of_samples);

    Integrate1(const F& f, const long total_number_of_samples, const long seed);

    void reset(const long total_number_of_samples)
    {
      assert(total_number_of_samples > 0);
      N_ = total_number_of_samples;

      running_total_ = {};
      sum_sigma_ = {};
    }

    void set_f(const F& f)
    {
      f_ = f;
    }

    void run_Monte_Carlo();

    Codomain running_total() const
    {
      return running_total_;
    }

    Codomain sum_sigma() const
    {
      return sum_sigma_;
    }

    // For debug purposes only

    //std::vector<Domain> domain_runs() const
    //{
    //  return domain_runs_;
    //}

    //std::vector<Codomain> codomain_runs() const
    //{
    //  return codomain_runs_;
    //}

  private:

    RandomNumberGenerator rng_;

    F f_;

    long N_;

    Codomain running_total_;
    Codomain sum_sigma_;

    // For debug purposes only

    //std::vector<Domain> domain_runs_;
    //std::vector<Codomain> codomain_runs_;
};

template <typename Domain, typename Codomain, class RandomNumberGenerator>
Integrate1<Domain, Codomain, RandomNumberGenerator>::Integrate1(
  const F& f,
  const long total_number_of_samples
  ):
  rng_{},
  f_{f},
  N_{total_number_of_samples},
  running_total_{},
  sum_sigma_{}
{
  assert(N_ > 0);
}

template <typename Domain, typename Codomain, class RandomNumberGenerator>
Integrate1<Domain, Codomain, RandomNumberGenerator>::Integrate1(
  const F& f,
  const long total_number_of_samples,
  const long seed
  ):
  rng_{seed},
  f_{f},
  N_{total_number_of_samples},
  running_total_{},
  sum_sigma_{}
{
  assert(N_ > 0);
}

template <class Domain, typename Codomain, class RandomNumberGenerator>
void Integrate1<Domain, Codomain, RandomNumberGenerator>::run_Monte_Carlo()
{
  for (int i {0}; i < N_; ++i)
  {
    // Partially for debug purposes only (can be combined)

    //const Domain x {rng_(&idum)};

    //const Codomain f_x {f_(x)};

    // For debug purposes only

    //domain_runs_.push_back(x);
    //codomain_runs_.push_back(f_x);

    const Codomain f_x {f_(rng_())};
    running_total_ += f_x;
    sum_sigma_ += f_x * f_x;
  }
}

} // namespace MonteCarlo

#endif // _MONTE_CARLO_INTEGRATE1_H_
