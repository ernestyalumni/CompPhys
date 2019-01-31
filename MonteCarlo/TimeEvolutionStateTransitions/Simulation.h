//------------------------------------------------------------------------------
/// \file Simulation.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Do the actual Monte Carlo routine.
/// \url https://en.wikipedia.org/wiki/Monte_Carlo_method
/// \ref CompPhys.pdf
/// \details A Monte Carlo simulation uses repeated sampling to obtain the
/// statistical properties of some phenomenon (or behavior).
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
#ifndef _MONTE_CARLO_TIME_EVOLUTION_STATE_TRANSITIONS_SIMULATION_H_
#define _MONTE_CARLO_TIME_EVOLUTION_STATE_TRANSITIONS_SIMULATION_H_

namespace MonteCarlo
{

namespace TimeEvolutionStateTransitions
{

class Simulation
{
  public:

    virtual void run() = 0;
}; // class Simulation

} // namespace TimeEvolutionStateTransitions

} // namespace MonteCarlo

#endif // _MONTE_CARLO_TIME_EVOLUTION_STATE_TRANSITIONS_SIMULATION_H_
