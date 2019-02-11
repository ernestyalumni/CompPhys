//------------------------------------------------------------------------------
/// \file Uniform_main.cu
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
/// nvcc -std=c++14 -lcurand -I ../../Utilities/ ../../Utilities/ErrorHandling.cpp Uniform_main.cu -o Uniform_main
//------------------------------------------------------------------------------
#include "Uniform.h"

#include <cstddef> // std::size_t 
#include <iostream>

using Curand::GenerateUniform;
using Curand::Uniform;
using Curand::UniformDistribution;

int main()
{
  // UniformDefaultConstructsWithcurandState_t
  {
    std::cout << "\n UniformDefaultConstructsWithcurandState_t \n";
    Uniform<64> uniform {16};
  }

  // UniformWorksWithcurandState_t
  {
    std::cout << "\n UniformWorksWithcurandState_t \n";

    // size or "length" L of the tuple or linear memory array.
    constexpr std::size_t L {128};
    // number of threads in a single thread block.
    constexpr std::size_t N_x {16};

    // Host input vectors
    float *h_sequence;

    h_sequence = (float*)malloc(L * sizeof(float));

    Uniform<L> uniform {N_x};

    uniform();

    {
      uniform.copy_to_host(h_sequence);

      for (int i {0}; i < 100; ++i)
      {
        std::cout << h_sequence[i] << ' ';
      }

      free(h_sequence);
    }
  }

  // UniformWorksWithcurandState_tAndDifferentSeeds
  {
    std::cout << "\n UniformWorksWithcurandState_tAndDifferentSeeds \n";

    // size or "length" L of the tuple or linear memory array.
    constexpr std::size_t L {128};
    // number of threads in a single thread block.
    constexpr std::size_t N_x {16};

    // Host input vectors
    float *h_sequence;

    h_sequence = (float*)malloc(L * sizeof(float));

    Uniform<L> uniform {N_x, 4321};

    {
      uniform.copy_to_host(h_sequence);

      for (int i {0}; i < 100; ++i)
      {
        std::cout << h_sequence[i] << ' ';
      }

      free(h_sequence);
    }
  }

  // UniformWorksMultipleTimes
  {
    std::cout << "\n UniformWorksMultipleTimes \n";

    // size or "length" L of the tuple or linear memory array.
    constexpr std::size_t L {128};
    // number of threads in a single thread block.
    constexpr std::size_t N_x {16};

    // Host input vectors
    float *h_sequence;

    h_sequence = (float*)malloc(L * sizeof(float));

    Uniform<L> uniform {N_x, 4321};
    {
      uniform.copy_to_host(h_sequence);

      for (int i {0}; i < 100; ++i)
      {
        std::cout << h_sequence[i] << ' ';
      }
    }

    {
      std::cout << "\n Second time \n";

      uniform();

      uniform.copy_to_host(h_sequence);

      for (int i {0}; i < 100; ++i)
      {
        std::cout << h_sequence[i] << ' ';
      }  
    }

    free(h_sequence);
  }


  // GenerateUniformDefaultConstructsWithcurandState_t
  {
    std::cout << "\n GenerateUniformDefaultConstructsWithcurandState_t \n";
//    GenerateUniform<64> generate_uniform {16};
  }

  // GenerateUniformWorksWithcurandState_t
  {
    std::cout << "\n GenerateUniformWorksWithcurandState_t \n";

    // size or "length" L of the tuple or linear memory array.
    constexpr std::size_t L {128};
    // number of threads in a single thread block.
//    constexpr std::size_t N_x {16};

    // Host input vectors
    float *h_sequence;

    h_sequence = (float*)malloc(L * sizeof(float));

  //  GenerateUniform<L> generate_uniform {N_x};

//    terminate called after throwing an instance of 'std::runtime_error'
//  what():  cudaErrorIllegalAddress an illegal memory access was encountered
//Aborted (core dumped)
// for this following command:
//    generate_uniform();

    {
  //    generate_uniform.copy_to_host(h_sequence);

    //  for (int i {0}; i < 100; ++i)
      //{
        //std::cout << h_sequence[i] << ' ';
      //}

      free(h_sequence);
    }
  }

  // UniformDistributionDefaultConstructsWithcurandState_t
  {
    std::cout << "\n UniformDistributionDefaultConstructsWithcurandState_t \n";
 
    //terminate called after throwing an instance of 'std::runtime_error'
    //  what():  cudaErrorIllegalAddress an illegal memory access was encountered
    //Aborted (core dumped)
//    UniformDistribution<64> uniform_distribution {16};
  }

}