//------------------------------------------------------------------------------
/// \file ParticlesInABox_main.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Particles in a box.
/// \url https://github.com/CompPhysics/ComputationalPhysics/blob/master/doc/Programs/LecturePrograms/programs/MCIntro/cpp/program2.cpp
/// \ref
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
///  g++ -std=c++17 -I ./ RandomNumberGenerators/RandomNumberGenerators.cpp ParticlesInABox_main.cpp -o ParticlesInABox_main
//------------------------------------------------------------------------------
#include "ParticlesInABox.h"
#include "RandomNumberGenerators/RandomNumberGenerators.h"

#include <iostream>
#include <vector>

using MonteCarlo::ParticlesInABox;
using MonteCarlo::RandomNumberGenerators::MinimalParkMiller;

int main()
{
  // ParticlesInABoxConstructsWithMinimalParkMiller
  {
    std::cout << "\n ParticlesInABoxConstructsWithMinimalParkMiller \n";

    ParticlesInABox<double, MinimalParkMiller> particles_in_a_box {1000, 4};
  }

  // ParticlesInABoxWithMinimalParkMillerWorks
  {
    std::cout << "\n ParticlesInABoxWithMinimalParkMillerWorks \n";

    std::vector<unsigned long> runs;

    ParticlesInABox<double, MinimalParkMiller> particles_in_a_box {1000, 4};

    particles_in_a_box.run();

    runs = particles_in_a_box.runs();

    for (unsigned long i {0}; i < 800; ++i)
    {
      std::cout << runs[i] << ' ';
    }
  }

}
