//------------------------------------------------------------------------------
/// \file Uniform.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Sequence of pseudorandom floats uniformly distributed between 0.0
/// and 1.0
/// \url https://docs.nvidia.com/cuda/curand/device-api-overview.htm
/// \ref 
/// \details Setup the initial states, wrapping curand_init() function.
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
///  nvcc -std=c++14 -lcurand -I ../../Utilities/ \
///   ../../Utilities/ErrorHandling.cpp InitialStates_main.cu -o \
///     InitialStates_main
//------------------------------------------------------------------------------
#ifndef _CURAND_UNIFORM_H_
#define _CURAND_UNIFORM_H_

#include "InitialStates.h"

#include "ErrorHandling.h"

#include <cstddef> // std::size_t
#include <curand_kernel.h>
#include <cuda_runtime_api.h> // blockIdx.x

namespace Curand
{

namespace Details
{

//--------------------------------------------------------------------------
/// \fn generate_uniform_kernel
/// \brief Function wrapper for curand_uniform
//--------------------------------------------------------------------------
template <typename StateType, std::size_t L>
__global__ void generate_uniform_kernel(StateType* states, float* sequence)
{
  // Get our global thread ID.
  std::size_t k_x {blockIdx.x * blockDim.x + threadIdx.x};

  // Make sure we don't go out of bounds.
  if (k_x >= L) { return; }  

  for (std::size_t tid {k_x}; tid < L; tid += gridDim.x * blockDim.x)
  {
    float x;

    // Copy state to local memory for efficiency
    StateType local_state = states[tid]; 

    // Generate pseudo-random uniform
    x = curand_uniform(&local_state);

    // Copy state back to global memory
    states[tid] = local_state;

    // Store results
    sequence[tid] = x;
  }
}

} // namespace Details

//------------------------------------------------------------------------------
/// \class Uniform
/// \brief Wrapper for curand_uniform
/// \details curand_uniform returns a sequence of pseudorandom floats uniformly
/// distributed between 0.0 and 1.0. It may return from 0.0 to 1.0, where 1.0 is
/// included and 0.0 excluded. Distribution functions may use any number of
/// unsigned integer values from a basic generator. The number of values
/// consumed is not guaranteed to be fixed.
/// \ref 3.1.4. Distributions, CUDA Toolkit Documentation, cuRAND
//------------------------------------------------------------------------------
template <
  std::size_t L,
  typename StateType = curandState_t
  >
class Uniform
{
  public:

    using HandleFree = Utilities::ErrorHandling::CUDA::HandleFree;
    using HandleMalloc = Utilities::ErrorHandling::CUDA::HandleMalloc;
    using HandleMemcpy = Utilities::ErrorHandling::CUDA::HandleMemcpy;
    using HandleMemset = Utilities::ErrorHandling::CUDA::HandleMemset;

    Uniform() = delete;

    //--------------------------------------------------------------------------
    /// \param default_N_x Default number of threads in a single block.
    //--------------------------------------------------------------------------
    Uniform(const std::size_t default_N_x, const unsigned long long seed);

    explicit Uniform(const std::size_t default_N_x);

    ~Uniform()
    {
      if (dev_sequence_ != nullptr)
      {
        const cudaError_t error {cudaFree(dev_sequence_)};
        HandleFree{}(error);
      }
    }

    //--------------------------------------------------------------------------
    /// \fn operator()()
    /// \brief Function call operator overload to generate uniform pseudo-random
    /// floating point numbers from 0.0 to 1.0, i.e. (0.0, 1.0]
    //--------------------------------------------------------------------------
    void operator()()
    {
      Details::generate_uniform_kernel<
        StateType,
        L>
        <<<default_M_x_, default_N_x_>>>(
          raw_states_.raw_states(), dev_sequence_);      
    }

    // Copy device memory to host
    void copy_to_host(float* h_data) const
    {
      const cudaError_t error {
        cudaMemcpy(
          h_data, dev_sequence_, L * sizeof(float), cudaMemcpyDeviceToHost)};
      
      HandleMemcpy{}(error);
    }

    float* dev_sequence()
    {
      return dev_sequence_;
    }

  private:

    // Default number of threads in a single thread block.
    const std::size_t default_N_x_;

    // Default number of thread blocks in a grid.
    const std::size_t default_M_x_;

    RawStates<L, StateType> raw_states_;

    float* dev_sequence_;
//    float* host_sequence_;

}; // class Uniform

template <std::size_t L, typename StateType>
Uniform<L, StateType>::Uniform(
  const std::size_t default_N_x,
  const unsigned long long seed
  ):
  default_N_x_{default_N_x},
  default_M_x_{(L + default_N_x_ - 1)/ default_N_x_},
  raw_states_{default_N_x, seed}
{
  // Allocate space for results on device.
  cudaError_t error {cudaMalloc((void**)&dev_sequence_, L * sizeof(float))};
  HandleMalloc{}(error);

  // Set results to 0
  error = cudaMemset(dev_sequence_, 0, L * sizeof(float));
  HandleMalloc{}(error);

  // Generate and use uniform pseudo-random
  Details::generate_uniform_kernel<StateType, L><<<default_M_x_, default_N_x_>>>(
    raw_states_.raw_states(), dev_sequence_);
}

template <std::size_t L, typename StateType>
Uniform<L, StateType>::Uniform(const std::size_t default_N_x):
  Uniform{default_N_x, 1234}
{}

} // namespace Curand

#endif // _CURAND_UNIFORM_H_