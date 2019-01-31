//------------------------------------------------------------------------------
/// \file RadioactiveDecays_main.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Radioactive decay, modeling first-order differential equations with
/// Monte Carlo methods.
/// \url https://github.com/CompPhysics/ComputationalPhysics/blob/master/doc/Programs/LecturePrograms/programs/MCIntro/cpp/program3.cpp
/// \ref 11.1.4 Radioactive Decay, Hjorth-Jensen (2015)
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
///  g++ -std=c++17 -I ./ RandomNumberGenerators/RandomNumberGenerators.cpp RadioactiveDecays_main.cpp -o RadioactiveDecays_main
//------------------------------------------------------------------------------
#include "RadioactiveDecays.h"
#include "RandomNumberGenerators/RandomNumberGenerators.h"

#include <iostream>
#include <vector>

using MonteCarlo::RadioactiveDecay::DecayOf1Nuclei;
using MonteCarlo::RadioactiveDecay::DecayOf2Nuclei;
using MonteCarlo::RadioactiveDecay::Details::NumbersOf2Types;
using MonteCarlo::RadioactiveDecay::TimeEvolution1;
using MonteCarlo::RandomNumberGenerators::MinimalParkMiller;

int main()
{
  // TimeEvolution1WorksWithMinimalParkMiller
  {
    std::cout << "\n TimeEvolution1WorksWithMinimalParkMiller \n";
    {
      TimeEvolution1<double, MinimalParkMiller> time_evolution_1 {0.5};

      for (int i {0}; i < 30; ++i)
      {
        std::cout << time_evolution_1() << ' ';
      }
    }
  }

  // DecayOf1NucleiConstructsWithMinimalParkMillerAndInitialConditions
  {
    std::cout <<
      "\n DecayOf1NucleiConstructsWithMinimalParkMillerAndInitialConditions \n";

    DecayOf1Nuclei<double, MinimalParkMiller> decay_of_1_nuclei {
      {0.1, 10000, 100}};
  }

  // DecayOf1NucleiWithMinimalParkMillerWorks
  {
    std::cout << "\n DecayOf1NucleiWithMinimalParkMillerWorks \n";

    std::vector<unsigned long> runs;

    DecayOf1Nuclei<double, MinimalParkMiller> decay_of_1_nuclei {
      {0.1, 100000, 1000}};

    decay_of_1_nuclei.run();

    runs = decay_of_1_nuclei.runs();

    for (unsigned long i {0}; i < 100; ++i)
    {
      std::cout << runs[i] << ' ';
    }
  }

  // DecayOf2NucleiWithMinimalParkMillerWorks
  {
    std::cout << "\n DecayOf2NucleiWithMinimalParkMillerWorks \n";

    std::vector<NumbersOf2Types> runs;

    DecayOf2Nuclei<double, MinimalParkMiller> decay_of_2_nuclei {
      {0.2, 0.1, 100000, 100000, 1000}};

    decay_of_2_nuclei.run();

    runs = decay_of_2_nuclei.runs();

    for (unsigned long i {0}; i < 100; ++i)
    {
      std::cout << runs[i].N_X_ << ',' << runs[i].N_Y_ << ' ';
    }
  }


  std::cout << std::endl;
}
