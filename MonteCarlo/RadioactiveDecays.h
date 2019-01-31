//------------------------------------------------------------------------------
/// \file RadioactiveDecays.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Radioactive decay, modeling first-order differential equations with
/// Monte Carlo methods.
/// \url
/// \ref 11.1.4 Radioactive Decay, Hjorth-Jensen (2015)
/// \details Radioactive decay.
/// There is public inheritance of the interface (interface inheritance),
/// separating the data collection and statistics from the actual Monte Carlo
/// run.
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
///  g++ -std=c++17 -I ./ RandomNumberGenerators/RandomNumberGenerators.cpp \
///   ParticlesInABox_main.cpp -o ParticlesInABox_main
//------------------------------------------------------------------------------
#ifndef _MONTE_CARLO_RADIOACTIVE_DECAYS_H_
#define _MONTE_CARLO_RADIOACTIVE_DECAYS_H_

#include "RandomNumberGenerators/RandomNumberGenerators.h"
#include "TimeEvolutionStateTransitions/Simulation.h"
#include "TimeEvolutionStateTransitions/TimeEvolution.h"

#include <cassert>
#include <vector>

namespace MonteCarlo
{

namespace RadioactiveDecay
{

namespace Details
{

struct NumbersOf2Types
{
  unsigned long N_X_;
  unsigned long N_Y_;
};

} // namespace Details

//------------------------------------------------------------------------------
/// \class TimeEvolution1
/// \brief Radioactive decay for a single type of nuclei ("X").
/// \details Recall that the probability of an event is a non-negative real
/// number; denote this set as RR.
//------------------------------------------------------------------------------
template <class RR, class RandomNumberGenerator>
class TimeEvolution1 : public TimeEvolutionStateTransitions::TimeEvolution<int>
{
  public:

    TimeEvolution1() = delete;

    explicit TimeEvolution1(const double& transition_probability):
      rng_{},
      transition_probability_{transition_probability}
    {
      assert(transition_probability >= 0.0 && transition_probability <= 1.0);
    }

    TimeEvolution1(const double& transition_probability, const long seed):
      rng_{seed},
      transition_probability_{transition_probability}
    {
      assert(transition_probability >= 0.0 && transition_probability <= 1.0);
    }

    int operator()()
    {
      return rng_() <= transition_probability_ ? 0 : 1;
    }

    int operator()(int& is_particle_there)
    {
      if (is_particle_there == 0)
      {
        return 0;
      }

//      return (rng_() <= transition_probability_) ? 0 : 1;
      return this->operator()();
    }

  private:

    RandomNumberGenerator rng_;

    double transition_probability_;
};

//------------------------------------------------------------------------------
/// \class TimeEvolution2
/// \brief Nucleus "X" decays to daughter nucleus "Y", which can also decay.
/// \details Recall that the probability of an event is a non-negative real
/// number; denote this set as RR.
//------------------------------------------------------------------------------
template <class RR, class RandomNumberGenerator>
class TimeEvolution2 :
  public TimeEvolutionStateTransitions::TimeEvolution<Details::NumbersOf2Types>
{
  public:

    using NumbersOf2Types = Details::NumbersOf2Types;

    TimeEvolution2() = delete;

    explicit TimeEvolution2(
      const double& transition_probability_X,
      const double& transition_probability_Y
      ):
      rng_{},
      transition_probability_X_{transition_probability_X},
      transition_probability_Y_{transition_probability_Y}
    {
      assert(
        transition_probability_X >= 0.0 && transition_probability_X <= 1.0);
      assert(
        transition_probability_Y >= 0.0 && transition_probability_Y <= 1.0);
    }

    TimeEvolution2(
      const double& transition_probability_X,
      const double& transition_probability_Y,
      const long seed
      ):
      rng_{seed},
      transition_probability_X_{transition_probability_X},
      transition_probability_Y_{transition_probability_Y}
    {
      assert(
        transition_probability_X >= 0.0 && transition_probability_X <= 1.0);
      assert(
        transition_probability_Y >= 0.0 && transition_probability_Y <= 1.0);
    }

    NumbersOf2Types operator()(NumbersOf2Types& are_particles_there)
    {
      NumbersOf2Types Y_created {0, 1};

      if (are_particles_there.N_X_ > 0)
      {
        return (rng_() <= transition_probability_X_) ?
          Y_created : are_particles_there;
      }

      if (are_particles_there.N_Y_ > 0)
      {
        return rng_() <= transition_probability_Y_ ?
          NumbersOf2Types{0, 0} : Y_created;
      }

      // TODO: Throw exception here

      return NumbersOf2Types{0, 0};
    }

  private:

    RandomNumberGenerator rng_;

    double transition_probability_X_;
    double transition_probability_Y_;
};

//------------------------------------------------------------------------------
/// \class DecayOf1Nuclei
//------------------------------------------------------------------------------
template <class RR, class RandomNumberGenerator>
class DecayOf1Nuclei : public TimeEvolutionStateTransitions::Simulation
{
  public:

    struct InitialConditions
    {
      double transition_probability_;
      unsigned long N_0_;
      unsigned long T_;
    };

    DecayOf1Nuclei() = delete;

    explicit DecayOf1Nuclei(const InitialConditions& initial_conditions):
      runs_{},
      time_evolution_{initial_conditions.transition_probability_},
      N_0_{initial_conditions.N_0_},
      N_t_{initial_conditions.N_0_},
      T_{initial_conditions.T_}
    {}

    DecayOf1Nuclei(
      const InitialConditions& initial_conditions,
      const long seed
      ):
      runs_{seed},
      time_evolution_{initial_conditions.transition_probability_},
      N_0_{initial_conditions.N_0_},
      N_t_{initial_conditions.N_0_},
      T_{initial_conditions.T_}
    {}

    void run()
    {
      for (unsigned long t {0}; t < T_; ++t)
      {
        unsigned long N_unstable {0};

        for (unsigned long n {0}; n < N_t_; ++n)
        {
          N_unstable += time_evolution_();
        }

        N_t_ = N_unstable;

        // Only for data recording.
        runs_.push_back(N_t_);
      }
    }

    std::vector<unsigned long> runs() const
    {
      return runs_;
    }

  private:

    // Only for data recording.
    std::vector<unsigned long> runs_;

    TimeEvolution1<RR, RandomNumberGenerator> time_evolution_;

    unsigned long N_0_;

    unsigned long N_t_;

    unsigned long T_;
};

//------------------------------------------------------------------------------
/// \class DecayOf1Nuclei
//------------------------------------------------------------------------------
template <class RR, class RandomNumberGenerator>
class DecayOf2Nuclei : public TimeEvolutionStateTransitions::Simulation
{
  public:

    using NumbersOf2Types = Details::NumbersOf2Types;

    struct InitialConditions
    {
      double transition_probability_X_;
      double transition_probability_Y_;
      unsigned long N_X_0_;
      unsigned long N_Y_0_;
      unsigned long T_;
    };

    DecayOf2Nuclei() = delete;

    explicit DecayOf2Nuclei(const InitialConditions& initial_conditions):
      runs_{},
      time_evolution_{
        initial_conditions.transition_probability_X_,
        initial_conditions.transition_probability_Y_
        },
      N_X_0_{initial_conditions.N_X_0_},
      N_Y_0_{initial_conditions.N_Y_0_},
      state_t_{initial_conditions.N_X_0_, initial_conditions.N_Y_0_},
      T_{initial_conditions.T_}
    {}

    DecayOf2Nuclei(
      const InitialConditions& initial_conditions,
      const long seed
      ):
      runs_{seed},
      time_evolution_{
        initial_conditions.transition_probability_X_,
        initial_conditions.transition_probability_Y_
        },
      N_X_0_{initial_conditions.N_X_0_},
      N_Y_0_{initial_conditions.N_Y_0_},
      state_t_{initial_conditions.N_X_0_, initial_conditions.N_Y_0_},
      T_{initial_conditions.T_}
    {}

    void run()
    {
      for (unsigned long t {0}; t < T_; ++t)
      {
        NumbersOf2Types state_unstable {0, 0};

        for (unsigned long n {0}; n < state_t_.N_X_; ++n)
        {
          NumbersOf2Types is_X_there {1, 0};
          is_X_there = time_evolution_(is_X_there);

          state_unstable.N_X_ += is_X_there.N_X_;
          state_unstable.N_Y_ += is_X_there.N_Y_;
        }

        for (unsigned long n {0}; n < state_t_.N_Y_; ++n)
        {
          NumbersOf2Types is_Y_there {0, 1};
          is_Y_there = time_evolution_(is_Y_there);

          state_unstable.N_Y_ += is_Y_there.N_Y_;
        }

        state_t_ = state_unstable;

        // Only for data recording.
        runs_.push_back(state_t_);
      }
    }

    std::vector<NumbersOf2Types> runs() const
    {
      return runs_;
    }

  private:

    // Only for data recording.
    std::vector<NumbersOf2Types> runs_;

    TimeEvolution2<RR, RandomNumberGenerator> time_evolution_;

    unsigned long N_X_0_;
    unsigned long N_Y_0_;

    NumbersOf2Types state_t_;

    unsigned long T_;
};

} // namespace RadioactiveDecay

} // namespace MonteCarlo

#endif // _MONTE_CARLO_RADIOIACTIVE_DECAYS_H_
