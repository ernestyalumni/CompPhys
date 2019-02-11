//------------------------------------------------------------------------------
/// \file InitialStates_main.cu
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Initial states for pseudorandom numbers main driver file.
/// \url https://docs.nvidia.com/cuda/curand/device-api-overview.html#device-api-overview
/// \ref 
/// \details Setup the initial states, wrapping curand_init() function.
/// Note that CUDA source files need to be named with the .cu suffix.
/// https://stackoverflow.com/questions/15288621/error-blockidx-was-not-declared-in-this-scope
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
/// nvcc -std=c++14 -lcurand -I ../../Utilities/ ../../Utilities/ErrorHandling.cpp InitialStates_main.cu -o InitialStates_main
//------------------------------------------------------------------------------
#include "InitialStates.h"

#include <cstddef> // std::size_t 
#include <iostream>

using Curand::Initialize;
using Curand::RawStates;

int main()
{
  // RawStatesDefaultConstructsWithcurandState_t
  {
    RawStates<64> raw_states {16};
  }

  // RawStatesDefaultConstructsWithcurandState_tAndGloballyInitializes
  {
    std::cout <<
      "\n RawStatesDefaultConstructsWithcurandState_tAndGloballyInitializes\n";
    RawStates<64> raw_states {16};

    raw_states.initialize();
  }


}