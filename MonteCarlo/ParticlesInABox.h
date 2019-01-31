//------------------------------------------------------------------------------
/// \file ParticleInABox.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Particles in a box.
/// \url
/// \ref
/// \details Particles in a box.
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
#ifndef _MONTE_CARLO_PARTICLES_IN_A_BOX_H_
#define _MONTE_CARLO_PARTICLES_IN_A_BOX_H_

#include "RandomNumberGenerators/RandomNumberGenerators.h"
#include "TimeEvolutionStateTransitions/Simulation.h"
#include "TimeEvolutionStateTransitions/TimeEvolution.h"

#include <cassert>
#include <vector>

namespace MonteCarlo
{

//------------------------------------------------------------------------------
/// \class ParticleTimeEvolution
/// \details Recall that the probability of an event is a non-negative real
/// number; denote this set as RR.
//------------------------------------------------------------------------------
template <class RR, class RandomNumberGenerator>
class ParticleTimeEvolution :
  public TimeEvolutionStateTransitions::TimeEvolution<unsigned long>
{
  public:

    ParticleTimeEvolution() = delete;

    //----------------------------------------------------------------------------
    /// \fn ParticleTimeEvolution
    /// \brief Constructor for a given total number of particles in a box.
    /// \details Box is partitioned into 2 partitions: left and right.
    //----------------------------------------------------------------------------
    explicit ParticleTimeEvolution(const unsigned long N):
      rng_{},
      N_{N}
    {}

    ParticleTimeEvolution(const unsigned long N, const long seed):
      rng_{seed},
      N_{N}
    {}

    unsigned long operator()(unsigned long& n_l)
    {
      assert(n_l <= N_);

      const RR pr {rng_()}; // pr = probability

      // boundary conditions
      if ((n_l != N_) && (n_l != 0))
      {
        return (static_cast<unsigned long>(pr * N_) <= n_l) ? (n_l - 1) :
          (n_l + 1);
      }
      else if (n_l == N_)
      {
        return n_l - 1;
      }
      else
      {
        return n_l + 1;
      }
    }

  private:

    RandomNumberGenerator rng_;

    unsigned long N_;
}; // class ParticleTimeEvolution

//------------------------------------------------------------------------------
/// \class ParticlesInABox
//------------------------------------------------------------------------------
template <class RR, class RandomNumberGenerator>
class ParticlesInABox : public TimeEvolutionStateTransitions::Simulation
{
  public:

    ParticlesInABox() = delete;

    ParticlesInABox(const unsigned long N, const unsigned long T):
      runs_{},
      time_evolution_{N},
      n_l_{N},
      N_{N},
      T_{T}
    {
      assert(N > 0);
      assert(T > 0);
    }

    ParticlesInABox(
      const unsigned long N,
      const unsigned long T,
      const long seed
      ):
      runs_{},
      time_evolution_{N, seed},
      n_l_{N},
      N_{N},
      T_{T}
    {
      assert(N > 0);
      assert(T > 0);
    }

    void run()
    {
      for (unsigned long t {0}; t < T_; ++t)
      {
        for (unsigned long n {0}; n < N_; ++n)
        {
          n_l_ = time_evolution_(n_l_);

          // Only for data recording.
          runs_.push_back(n_l_);
        }
      }
    }

    std::vector<unsigned long> runs() const
    {
      return runs_;
    }

  private:

    // Only for data recording.
    std::vector<unsigned long> runs_;

    ParticleTimeEvolution<RR, RandomNumberGenerator> time_evolution_;

    unsigned long n_l_;

    unsigned long N_;

    unsigned long T_;
};

} // namespace MonteCarlo

#endif // _MONTE_CARLO_PARTICLES_IN_A_BOX_H_
