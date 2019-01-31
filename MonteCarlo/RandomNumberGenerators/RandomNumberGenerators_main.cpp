//------------------------------------------------------------------------------
/// \file RandomNumberGenerators_main.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Random number generators (RNG) main driver file.
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
///  g++ -std=c++17 RandomNumberGenerators.cpp RandomNumberGenerators_main.cpp -o RandomNumberGenerators_main
//------------------------------------------------------------------------------
#include "RandomNumberGenerators.h"

#include <functional>
#include <iostream>
#include <memory>

using MonteCarlo::RandomNumberGenerators::BaysDurhamShuffle;
using MonteCarlo::RandomNumberGenerators::Details::MASK;
using MonteCarlo::RandomNumberGenerators::LEcuyer;
using MonteCarlo::RandomNumberGenerators::MinimalParkMiller;
using MonteCarlo::RandomNumberGenerators::Uniform;

double func(double x)
{
  double value;
  value = 4 / (1. + x * x);

  return value;
} // end of function to evaluate

float f_as_float(float x)
{
  return {4 / (1.0f + x * x)};
} // end of function to evaluate

/// \url https://en.cppreference.com/w/cpp/utility/functional/function
/// \details Class template std::function is a general-purpose polymorphic
/// function wrapper, instances of std::function can store, copy, and invoke
/// any Callable target - functions, lambda expressions, bind expression, or
/// other function objects, as well as pointers to member functions and pointers
/// to data members.

template <typename Domain, typename Codomain>
std::function<Codomain(Domain)> f;

template <typename Domain, typename Codomain>
class Function : public std::function<Codomain(Domain)>
{
  public:

    using std::function<Codomain(Domain)>::function;
};

int main()
{

  {
    long* raw_pointer_long;
    std::cout << " raw_pointer_long : " << raw_pointer_long << '\n'; // 0x4010d0
    std::cout << " &raw_pointer_long : " << &raw_pointer_long << '\n';
      // 0x7fffcf006848

    // Segmentation fault (core dumped)
//    std::cout << " *raw_pointer_long : " << *raw_pointer_long << '\n';

    std::cout << sizeof(long*) << '\n'; // 8

    std::unique_ptr<long> uptr_long;

    std::cout << sizeof(std::unique_ptr<long>) << '\n'; // 8

    // error no known conversion
//    std::cout << "\n uptr_long : " << uptr_long << '\n';
    std::cout << "\n &uptr_long : " << &uptr_long << '\n'; // 0x7fffcf006840

    // Segmentation fault
//    std::cout << "\n *uptr_long : " << *uptr_long << '\n';

    // \url https://en.cppreference.com/w/cpp/memory/unique_ptr
    // get, returns pointer to the managed object
    std::cout << "\n uptr_long.get() : " << uptr_long.get() << '\n'; // 0

    // Segmentation fault (core dumped)
//    std::cout << "\n *uptr_long.get() : " << *uptr_long.get() << '\n'; // 0

    // error: lvalue required as unary & operand
    //std::cout << "\n &(uptr_long.get()) : " << &(uptr_long.get()) << '\n';

    // double free or corruption (out), Aborted (core dumped)
//    long* raw_pointer_long_to_unique;
  //  std::unique_ptr<long> uptr_long_to_raw {raw_pointer_long_to_unique};

//    std::unique_ptr<long> uptr_long_to_raw {
  //    std::make_unique<long>(&raw_pointer_long)};

    // double free or corruption (out), Aborted (core dumped)
//    std::unique_ptr<long> uptr_long_to_raw (std::move(raw_pointer_long));

  }

  // \url https://en.cppreference.com/w/cpp/utility/functional/function
  std::cout << " \n std::function playground \n";
  {
    std::function<float(float)> f_test;

    // store a free function
    // operator = assigns a new target
    f_test = f_as_float;

    std::cout << f_test(5) << '\n';
  }

  {
    f<float, float>;

    f<float, float> = f_as_float;

    std::cout << f<float, float>(5) << '\n';
  }

  {
    Function<float, float> function_float_to_float;
    function_float_to_float = f_as_float;
    std::cout << function_float_to_float(5) << '\n';
  }

  // MinimalParkMillerPlayground
  std::cout << "\n MinimalParkMillerPlayground \n";
  {
    long idum {-1};

    std::cout << " idum : " << idum << " &idum : " << &idum << '\n';

    std::cout << " MASK : " << MASK << '\n';

    std::cout << std::hexfloat << MASK << ' ' << std::hex << MASK << //std::dec <<
      std::defaultfloat << '\n';

    long* idum_test {&idum};

    std::cout << " idum_test : " << idum_test << " *idum_test : " << *idum_test
      << std::dec << ' ' << *idum_test << " &idum_test : " <<
        &idum_test << '\n';

    *idum_test ^= MASK;

    std::cout << "\n After bit XOR with MASK \n ";

    std::cout << " idum : " << idum << " &idum : " << &idum << '\n';

    std::cout << " idum_test : " << idum_test << " *idum_test : " << std::hex <<
      *idum_test << std::dec << ' ' << *idum_test << " &idum_test : " <<
        &idum_test << '\n';

    std::cout << "\n Check the steps of MinimalParkMiller \n";

    {
      long idum_test {-1};

      // ran0(&idum_test) where ran0(long* idum)

      std::cout << " idum_test : " << idum_test << " &idum_test : " <<
        &idum_test << '\n';

      idum_test ^= MASK;

      std::cout << " idum_test : " << idum_test << " &idum_test : " <<
          &idum_test << '\n';


      long k {(idum_test) / MonteCarlo::RandomNumberGenerators::Details::IQ};
      std::cout << " k : " << k << '\n';

      idum_test = MonteCarlo::RandomNumberGenerators::Details::IA *
        (idum_test - k * MonteCarlo::RandomNumberGenerators::Details::IQ) -
          MonteCarlo::RandomNumberGenerators::Details::IR * k;

      std::cout << " idum_test : " << idum_test << " &idum_test : " <<
          &idum_test << '\n';

      if (idum_test < 0)
      {
        idum_test += MonteCarlo::RandomNumberGenerators::Details::IM;
      }

      std::cout << " idum_test : " << idum_test << " &idum_test : " <<
          &idum_test << '\n';

      double ans {MonteCarlo::RandomNumberGenerators::Details::AM * (idum_test)};

      std::cout << " ans : " << ans << '\n';

      idum_test ^= MASK;

      std::cout << " idum_test : " << idum_test << " &idum_test : " <<
          &idum_test << '\n';

      std::cout << " ans : " << ans << '\n';
    }

    std::cout << "\n Check, for a unique ptr, the steps of MinimalParkMiller \n";

    {
      std::unique_ptr<long> u_ptr_seed {std::make_unique<long>(-1)};

      // cannot convert ‘u_ptr_seed’ (type ‘std::unique_ptr<long int>’) to type
      // ‘const unsigned char*’

      std::cout << " &u_ptr_seed " << &u_ptr_seed << " *u_ptr_seed : " <<
        *u_ptr_seed << " u_ptr_seed.get() : " << u_ptr_seed.get() <<
        " *u_ptr_seed.get() : " << *u_ptr_seed.get() << " *u_ptr_seed : " <<
        *u_ptr_seed << '\n';

      *u_ptr_seed ^= MASK;

      std::cout << " &u_ptr_seed " << &u_ptr_seed << " *u_ptr_seed : " <<
        *u_ptr_seed << " u_ptr_seed.get() : " << u_ptr_seed.get() <<
        " *u_ptr_seed.get() : " << *u_ptr_seed.get() << " *u_ptr_seed : " <<
        *u_ptr_seed << '\n';

      long k {(*u_ptr_seed) / MonteCarlo::RandomNumberGenerators::Details::IQ};
      std::cout << " k : " << k << '\n';

      *u_ptr_seed = MonteCarlo::RandomNumberGenerators::Details::IA *
        (*u_ptr_seed - k * MonteCarlo::RandomNumberGenerators::Details::IQ) -
          MonteCarlo::RandomNumberGenerators::Details::IR * k;

      std::cout << " &u_ptr_seed " << &u_ptr_seed << " *u_ptr_seed : " <<
        *u_ptr_seed << " u_ptr_seed.get() : " << u_ptr_seed.get() <<
        " *u_ptr_seed.get() : " << *u_ptr_seed.get() << " *u_ptr_seed : " <<
        *u_ptr_seed << '\n';

      if (*u_ptr_seed < 0)
      {
        *u_ptr_seed += MonteCarlo::RandomNumberGenerators::Details::IM;
      }

      std::cout << " &u_ptr_seed " << &u_ptr_seed << " *u_ptr_seed : " <<
        *u_ptr_seed << " u_ptr_seed.get() : " << u_ptr_seed.get() <<
        " *u_ptr_seed.get() : " << *u_ptr_seed.get() << " *u_ptr_seed : " <<
        *u_ptr_seed << '\n';

      double ans {MonteCarlo::RandomNumberGenerators::Details::AM * (*u_ptr_seed)};

      std::cout << " ans : " << ans << '\n';

      *u_ptr_seed ^= MASK;

      std::cout << " &u_ptr_seed " << &u_ptr_seed << " *u_ptr_seed : " <<
        *u_ptr_seed << " u_ptr_seed.get() : " << u_ptr_seed.get() <<
        " *u_ptr_seed.get() : " << *u_ptr_seed.get() << " *u_ptr_seed : " <<
        *u_ptr_seed << '\n';

      std::cout << " ans : " << ans << '\n';
    }
  }

  // MinimalParkMillerInterface
  std::cout << "\n MinimalParkMillerInterface \n";
  {
    MinimalParkMiller minimal_random_number;

    long idum {-1};

    for (int i {0}; i < 20; ++i)
    {
      std::cout << std::defaultfloat << minimal_random_number() << ' ' <<
        idum << ' ';
    }
  }

  // MinimalParkMillerWorksMultipleTimesWithSameSeedValue
  std::cout << "\n MinimalParkMillerWorksMultipleTimesSameSeedValue \n";
  {
    for (int i {0}; i < 30; ++i)
    {
      long idum {-1};
      {
        MinimalParkMiller minimal_random_number {idum};
        std::cout << minimal_random_number() << ' ';
      }
    }
  }

  // MinimalParkMillerWorksMultipleTimesWithDifferentSeedValue
  std::cout <<
    "\n MinimalParkMillerWorksMultipleTimesWithDifferentSeedValue \n";
  {
    MinimalParkMiller minimal_random_number;

    long idum {-1};

    for (int i {0}; i < 30; ++i)
    {
      MinimalParkMiller minimal_random_number {idum + i};
      std::cout << minimal_random_number() << ' ';
    }
  }

  // BaysDurhamShuffleWorks
  std::cout << " \n BaysDurhamShuffle \n";
  {
    BaysDurhamShuffle bays_durham_number;

    long idum {-1};

    for (int i {0}; i < 30; ++i)
    {
      std::cout << bays_durham_number() << ' ' << idum << ' ';
    }
  }
  // LEcuyerWorks
  std::cout << "\n LecuyerWorks\n";
  {
    LEcuyer lecuyer;

    long idum {-1};

    for (int i {0}; i < 30; ++i)
    {
      std::cout << lecuyer() << ' ' << idum << ' ';
    }
  }

  // UniformWorks
  std::cout << "\n UniformWorks\n";
  {
    Uniform uniform;

    long idum {-1};

    for (int i {0}; i < 30; ++i)
    {
      std::cout << uniform() << ' ' << idum << ' ';
    }
  }

}
