//------------------------------------------------------------------------------
/// \file IntegratePi_main.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Integrate to compute pi.
/// \ref pp. 345, program1.cpp, 11.1 Introduction, Morten Hjorth-Jensen,
/// Computational Physics, Lecture Notes Fall 2015, Dept. of Physics, U. of
/// Oslo, Ch. 11. Outline of the Monte Carlo Strategy
/// \url https://github.com/CompPhysics/ComputationalPhysics/blob/master/doc/Programs/LecturePrograms/programs/MCIntro/cpp/program1.cpp
/// \details Ring, following interface implementation.
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
///  g++ -std=c++17 Tuple2_main.cpp -o Tuple2_main
//------------------------------------------------------------------------------
#include <cmath>
#include <iostream>

constexpr long MASK = 123459876;
constexpr long IQ = 127773;
constexpr long IA = 16807;
constexpr long IR = 2836;
constexpr long IM = 2147483647;
constexpr double AM = {1.0 / static_cast<double>(IM)};

double ran0(long *idum)
{
  long k;
  double ans;

  *idum ^= MASK;
  k = (*idum) / IQ;
  *idum = IA * (*idum - k * IQ) - IR * k;

  if (*idum < 0)
  {
    *idum += IM;
  }

  ans = AM * (*idum);
  *idum ^= MASK;

  return ans;
}


//------------------------------------------------------------------------------
/// Here we defined various functions called by the main program
/// this function defines the function to integrate
//------------------------------------------------------------------------------
double func(double x);

//------------------------------------------------------------------------------
/// Main function begins here
/// \details The algorithm is as follows:
/// * Choose the number of Monte Carlo samples N.
/// * Perform a loop over N and for each step generate a random number x, in the
/// interval [0, 1] through a call to a random number generator.
/// * Use this number to evaluate f(x_i)
/// * Evaluate the contributions to the mean value and the standard deviation
/// for each loop.
/// * After N samples, calculate the final mean value and the standard
/// deviation.
//------------------------------------------------------------------------------

int main()
{
  int i, n;
  long idum;
  double crude_mc, x, sum_sigma, fx, variance, exact;

  std::cout << "\n Read in the number of Monte-Carlo samples \n";
  std::cin >> n;

  crude_mc = sum_sigma = 0.;
  idum = -1;

  exact = acos(-1.);

  // evaluate the integral with a crude Monte-Carlo method
  for (int i {1}; i <= n; ++i)
  {
    x = ran0(&idum);
    fx = func(x);
    crude_mc += fx;
    sum_sigma += fx * fx;
  }

  crude_mc = crude_mc/((double) n);
  sum_sigma = sum_sigma / ((double) n);
  variance = sum_sigma - crude_mc * crude_mc;

  // final output

  std::cout << " variance = " << variance << " Integral = " << crude_mc <<
    " Exact= " << M_PI << std::endl;

} // end of main program

//------------------------------------------------------------------------------
/// \brief This function defines the function to integrate
//------------------------------------------------------------------------------
double func(double x)
{
  double value;
  value = 4 / (1. + x * x);

  return value;
} // end of function to evaluate



