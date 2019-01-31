//------------------------------------------------------------------------------
/// \file Integrate1_main.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Random number generators (RNG).
/// \url https://github.com/CompPhysics/ComputationalPhysics/blob/master/doc/Programs/LecturePrograms/programs/MCIntro/cpp/program1.cpp
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
///  g++ -I ../ -std=c++17 RandomNumberGenerators/RandomNumberGenerators.cpp Integrate1_main.cpp -o Integrate1_main
//------------------------------------------------------------------------------
#include "Integrate1.h"

#include "RandomNumberGenerators/RandomNumberGenerators.h"
#include "Rings/Function.h"

#include <iostream>

using MonteCarlo::Integrate1;
using MonteCarlo::RandomNumberGenerators::MinimalParkMiller;
using Rings::Function;

const double func(const double x)
{
  return 4 / (1.0 + x * x);
} // end of function to evaluate

int main()
{

  // MinimalParkMillerWorks
  {
    std::cout << "\n MinimalParkMiller \n";
    MinimalParkMiller minimal_park_miller;

    long idum {-1};

    for (int i {0}; i < 30; ++i)
    {
      std::cout << minimal_park_miller() << ' ';
    }
  }

  // Integrate1Constructs
  {
    std::cout << "\n Integrate1Constructs \n";

    Integrate1<double, double, MinimalParkMiller> integrate1 {func, 2};
  }

  // Integrate1Works
  {
//    Function<double, double> f_1 {func};
    const long total_number_of_samples {2};

    Integrate1<double, double, MinimalParkMiller> integrate1 {
      func, total_number_of_samples};

    integrate1.run_Monte_Carlo();

    std::cout << " running_total : " << integrate1.running_total() <<
      " sum_sigma : " << integrate1.sum_sigma() << '\n';

    // For debug purposes only

    //for (auto& x : integrate1.domain_runs())
    //{
    //  std::cout << x << ' ';
    //}
    //std::cout << "\n codomain_runs : \n ";
    //for (auto& x : integrate1.codomain_runs())
    //{
    //  std::cout << x << ' ';
    //}

  }

  // Integrate1WorksForMultipleSamples
  {
    std::cout << "\n Integrate1WorksForMultipleSamples \n";

    long total_number_of_samples {30};

    Integrate1<double, double, MinimalParkMiller> integrate1 {
      func, total_number_of_samples};

    integrate1.run_Monte_Carlo();

    std::cout << " running_total : " << integrate1.running_total() <<
      " sum_sigma : " << integrate1.sum_sigma() << '\n';

    //for (auto& x : integrate1.domain_runs())
    //{
    //  std::cout << x << ' ';
    //}
    //std::cout << "\n codomain_runs : \n ";
    //for (auto& x : integrate1.codomain_runs())
    //{
    //  std::cout << x << ' ';
    //}
  }

  // Integrate1WorksForDifferentNumberOfSamples
  {
    std::cout << "\n Integrate1WorksForDifferentNumberOfSamples \n";

    long total_number_of_samples {10};

    Integrate1<double, double, MinimalParkMiller> integrate1 {
      func, total_number_of_samples};

    integrate1.run_Monte_Carlo();

    std::cout << "\n total_number_of_samples " << total_number_of_samples <<
      " running_total : " << integrate1.running_total() << " sum_sigma : " <<
        integrate1.sum_sigma() << '\n';

    total_number_of_samples = 100;

    integrate1.reset(total_number_of_samples);
    integrate1.run_Monte_Carlo();

    std::cout << "\n total_number_of_samples " << total_number_of_samples <<
      " running_total : " << integrate1.running_total() << " sum_sigma : " <<
        integrate1.sum_sigma() << '\n';

    total_number_of_samples = 1000;

    integrate1.reset(total_number_of_samples);
    integrate1.run_Monte_Carlo();

    std::cout << "\n total_number_of_samples " << total_number_of_samples <<
      " running_total : " << integrate1.running_total() << " sum_sigma : " <<
        integrate1.sum_sigma() << '\n';

    total_number_of_samples = 10000;

    integrate1.reset(total_number_of_samples);
    integrate1.run_Monte_Carlo();

    std::cout << "\n total_number_of_samples " << total_number_of_samples <<
      " running_total : " << integrate1.running_total() << " sum_sigma : " <<
        integrate1.sum_sigma() << '\n';

  }
}
