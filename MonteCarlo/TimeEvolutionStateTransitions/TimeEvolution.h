//------------------------------------------------------------------------------
/// \file TimeEvolutionStateTransition.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Discrete time evolution for a state of finite number of state
/// variables.
/// \url
/// \ref CompPhys.pdf
/// \details Discrete time evolution for a system characterized by a finite
/// number of state variables.
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
#ifndef _MONTE_CARLO_TIME_EVOLUTION_STATE_TRANSITIONS_TIME_EVOLUTION_H_
#define _MONTE_CARLO_TIME_EVOLUTION_STATE_TRANSITIONS_TIME_EVOLUTION_H_

namespace MonteCarlo
{

namespace TimeEvolutionStateTransitions
{

template <class States>
class TimeEvolution
{
  public:

    virtual States operator()(States&) = 0;
}; // class TimeEvolution

} // namespace TimeEvolutionStateTransitions

} // namespace MonteCarlo

#endif // _MONTE_CARLO_TIME_EVOLUTION_STATE_TRANSITIONS_TIME_EVOLUTION_H_
