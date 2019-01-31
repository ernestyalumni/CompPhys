//------------------------------------------------------------------------------
/// \file RandomNumberGenerators.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Random number generators (RNG).
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
///  g++ -std=c++17 RandomNumberGenerators.cpp RandomNumberGenerators_main.cpp \
///   -o RandomNumberGenerators_main
//------------------------------------------------------------------------------
#include "RandomNumberGenerators.h"

namespace MonteCarlo
{

namespace RandomNumberGenerators
{

MinimalParkMiller::MinimalParkMiller():
  seed_{std::make_unique<long>(-1)}
{}

MinimalParkMiller::MinimalParkMiller(const long seed):
  seed_{std::make_unique<long>(seed)}
{}

double MinimalParkMiller::operator()()
{
  *seed_ ^= MASK_;
  long k {(*seed_) / IQ_};
  *seed_ = IA_ * (*seed_ - k * IQ_) - IR_ * k;

  if (*seed_ < 0)
  {
    *seed_ += IM_;
  }

  double ans {AM_ * (*seed_)};
  *seed_ ^= MASK_;

  return ans;
}

//double MinimalParkMiller::operator()(const long seed)
//{
  // \brief Replaces the managed object.
  // \url https://en.cppreference.com/w/cpp/memory/unique_ptr/reset
//  seed_.reset(std::make_unique<long>(seed).get());

//  return this->operator()();
//}

BaysDurhamShuffle::BaysDurhamShuffle():
  MinimalParkMiller{}
{}

BaysDurhamShuffle::BaysDurhamShuffle(const long seed):
  MinimalParkMiller{seed}
{}

//double BaysDurhamShuffle::operator()(long* idum)
double BaysDurhamShuffle::operator()()
{
  int j;
  long k;

  static long iy {0};
  static long iv[NTAB_];

  double temp;

  if (*seed() <= 0 || !iy)
  {
    (-(*seed()) < 1) ? *seed() = 1 : *seed() = -(*seed());

    for (j = NTAB_ + 7; j >= 0; j--)
    {
      k = (*seed()) / IQ_;
      *seed() = IA_ * (*seed() - k * IQ_) - IR_ * k;
      if (*seed() < 0)
      {
        *seed() += IM_;
      }

      if (j < NTAB_)
      {
        iv[j] = *seed();
      }
    }
    iy = iv[0];
  }

  k = (*seed()) / IQ_;
  *seed() = IA_ * (*seed() - k * IQ_) - IR_ * k;

  if (*seed() < 0)
  {
    *seed() += IM_;
  }

  j = iy / NDIV_;
  iy = iv[j];
  iv[j] = *seed();

  return ((AM_* iy) > RNMX_) ? RNMX_ : AM_ * iy;
}

LEcuyer::LEcuyer():
  BaysDurhamShuffle{}
{}

LEcuyer::LEcuyer(const long seed):
  BaysDurhamShuffle{seed}
{}

double LEcuyer::operator()()
{
  int j;
  long k;
  static long idum2 {123456789};
  static long iv[NTAB_];

  static long iy {0};

  if (*seed() <= 0)
  {
    (-(*seed()) < 1) ? *seed() = 1 : *seed() = (-(*seed()));

    idum2 = (*seed());

    for (j = NTAB_ + 7; j >= 0; j--)
    {
      k = (*seed()) / IQ1_;
      *seed() = IA1_ * (*seed() - k * IQ1_) - k * IR1_;

      if (*seed() < 0)
      {
        *seed() += IM1_;
      }

      if (j < NTAB_)
      {
        iv[j] = *seed();
      }
    }
    iy = iv[0];
  }

  k = (*seed()) / IQ1_;

  *seed() = IA1_ * (*seed() - k * IQ1_) - k * IR1_;

  if (*seed() < 0)
  {
    *seed() += IM1_;
  }

  k = idum2 / IQ2_;

  idum2 = IA2_ * (idum2 - k * IQ2_) - k * IR2_;

  if (idum2 < 0)
  {
    idum2 += IM2_;
  }

  j = iy / NDIV_;

  iy = iv[j] - idum2;

  iv[j] = *seed();

  if (iy < 1)
  {
    iy += IMM1_;
  }

  return ((AM1_ * iy) > RNMX_) ? RNMX_ : AM1_ * iy;
}

Uniform::Uniform():
  seed_{std::make_unique<long>(-1)}
{}

Uniform::Uniform(const long seed):
  seed_{std::make_unique<long>(seed)}
{}

double Uniform::operator()()
//double Uniform::operator()()
{
  static int inext, inextp;
  static long ma[56]; // value 56 is special, do not modify
  static int iff {0};

  long mj;

  if (*seed_ < 0 || iff == 0) // initialization
  {
    iff = 1;

    mj = MSEED_ - (*seed_ < 0 ? -*seed_ : *seed_);
    mj %= MBIG_;
    ma[55] = mj; // initialize ma[55]

    for (int i {1}, mk {1}; i <= 54; i++) // initialize rest of table
    {
      int ii {(21 * i) % 55};
      ma[ii] = mk;
      mk = mj - mk;
      if (mk < MZ_)
      {
        mk += MBIG_;
      }
      mj = ma[ii];
    }

    for (int k {0}; k < 4; ++k)
    {
      // randomize by "warming up" the generator
      for (int i {1}; i <= 55; ++i)
      {
        ma[i] -= ma[1 + (i + 30) % 55];
        if (ma[i] < MZ_)
        {
          ma[i] += MBIG_;
        }
      }
    }

    inext = 0; // prepare indices for first generator number
    inextp = 31; // 31 is special
    *seed_ = 1;
  }

  if (++inext == 56)
  {
    inext = 1;
  }

  if (++inextp == 56)
  {
    inextp = 1;
  }

  mj = ma[inext] - ma[inextp];

  if (mj < MZ_)
  {
    mj += MBIG_;
  }

  ma[inext] = mj;

  return mj * FAC_;
}

} // namespace RandomNumberGenerators

} // namespace MonteCarlo
