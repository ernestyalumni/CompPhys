//------------------------------------------------------------------------------
/// \file RadioactiveDecays.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Radioactive decay, modeling first-order differential equations with
/// Monte Carlo methods.
/// \url
/// \ref 11.1.4 Radioactive Decay, Hjorth-Jensen (2015)
/// \details Particles in a box.
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
#ifndef _MONTE_CARLO_PARTICLES_IN_A_BOX_H_
#define _MONTE_CARLO_PARTICLES_IN_A_BOX_H_

#include "RandomNumberGenerators/RandomNumberGenerators.h"
#include "TimeEvolutionStateTransitions/TimeEvolution.h"

#include <cassert>
#include <vector>

namespace MonteCarlo
{

namespace RadioactiveDecay
{

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

    int operator()()
    {
      long idum {-1};
      return rng_(&idum) <= transition_probability_ ? 0 : 1;
    }

    int operator()(int& is_particle_there)
    {
      if (is_particle_there == 0)
      {
        return 0;
      }

      long idum {-1};
      return (rng_(&idum) <= transition_probability_) ? 0 : 1;
    }

  private:

    RandomNumberGenerator rng_;

    double transition_probability_;
};

//------------------------------------------------------------------------------
/// \class DecayOf1Nuclei
//------------------------------------------------------------------------------
template <class RR, class RandomNumberGenerator>
class DecayOf1Nuclei
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

} // namespace RadioactiveDecay

} // namespace MonteCarlo

#endif // _MONTE_CARLO_PARTICLES_IN_A_BOX_H_
