//------------------------------------------------------------------------------
/// \file InitialStates.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Initial states for pseudorandom numbers..
/// \url https://docs.nvidia.com/cuda/curand/device-api-overview.html
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
#ifndef _CURAND_INITIAL_STATES_H_
#define _CURAND_INITIAL_STATES_H_

#include "ErrorHandling.h"

#include <cstddef> // std::size_t
#include <curand_kernel.h>
#include <cuda_runtime_api.h> // blockIdx.x
#include <type_traits>

namespace Curand
{

namespace Details
{

//--------------------------------------------------------------------------
/// \fn setup_kernel
/// \brief Function wrapper for curand_init
/// \details Recall that 
/// 
/// __device__ void
/// curand_init(
///   unsigned long long seed, unsigned long long sequence,
///   unsigned long long offset, curandState_t* state)
/// Also note taht this error would be obtained if this was made a class
/// member function:
///
/// warning: inline qualifier ignored for "__global__" function
/// error: illegal combination of memory qualifiers
/// 
//--------------------------------------------------------------------------
template <typename StateType, std::size_t L>
__global__ void setup_kernel(
  StateType* states,
  unsigned long long seed = 1234,
  unsigned long long offset = 0)
{
  // Get our global thread ID.
  std::size_t k_x {blockIdx.x * blockDim.x + threadIdx.x};

  // Make sure we don't go out of bounds.
  if (k_x >= L) { return; }  

  for (std::size_t tid {k_x}; tid < L; tid += gridDim.x * blockDim.x)
  {
    curand_init(seed, tid, offset, &states[tid]);
  }
}

} // namespace Details

//------------------------------------------------------------------------------
/// \class
/// \details Use RAII (Resource Acquisition Is Initialization) for a raw pointer
/// to the initial states, i.e. pnrg (pseudorandom number generator) states
/// \tparam State Parameter is meant to be 
//------------------------------------------------------------------------------
template <
  std::size_t L,
  typename StateType = curandState_t,
  typename std::enable_if_t<
      (std::is_same<StateType, curandState_t>::value ||
        std::is_same<StateType, curandStatePhilox4_32_10_t>::value ||
          std::is_same<StateType, curandStateMRG32k3a>::value ||
            std::is_same<StateType, curandState>::value)>* = nullptr
  >
class RawStates
{
  public:

    using HandleFree = Utilities::ErrorHandling::CUDA::HandleFree;
    using HandleMalloc = Utilities::ErrorHandling::CUDA::HandleMalloc;

    RawStates() = delete;

    //--------------------------------------------------------------------------
    /// \fn Constructor with default seed.
    /// \param default_N_x Default number of threads in a single block.
    /// \brief Constructor with default seed; note that states are not
    /// initialized; must be initialized manually.
    /// \details EY (20190202) I would like to know how to write this ctor
    /// with the correct template arguments, outside the class template
    /// definition. I tried to Google search for the answer; I got this
    /// https://stackoverflow.com/questions/41904099/why-sfinae-enable-if-works-from-inside-class-definition-but-not-from-outside
    /// but I don't think it answers my question.
    //--------------------------------------------------------------------------
    explicit RawStates(const std::size_t default_N_x):
      default_N_x_{default_N_x},
      default_M_x_{(L + default_N_x_ - 1)/ default_N_x_},
      seed_{1234}
    {
      const cudaError_t error {
        cudaMalloc((void**)&raw_states_, L * sizeof(StateType))};
      HandleMalloc{}(error);

      Details::setup_kernel<StateType, L><<<default_M_x_, default_N_x_>>>(
        raw_states_);
    }

    //--------------------------------------------------------------------------
    /// \fn Constructor with variable seed.
    /// \brief Constructor with default seed; note that states are not
    /// initialized; must be initialized manually.
    //--------------------------------------------------------------------------
    RawStates(const std::size_t default_N_x, const unsigned long long seed):
      default_N_x_{default_N_x},
      default_M_x_{(L + default_N_x_ - 1)/ default_N_x_},
      seed_{seed}
    {
      const cudaError_t error {cudaMalloc((void**)&raw_states_, L)};
      HandleMalloc{}(error);

      Details::setup_kernel<StateType, L><<<default_M_x_, default_N_x_>>>(
        raw_states_, seed);
    }

    // Not movable, not copyable.
    RawStates(const RawStates&) = delete; // copy ctor
    RawStates(RawStates&&) = delete; // move ctor
    RawStates& operator=(const RawStates&) = delete; // copy assignment ctor
    RawStates& operator=(RawStates&&) = delete; // move assignment ctor

    ~RawStates()
    {
      if (raw_states_ != nullptr)
      {
        const cudaError_t error {cudaFree(raw_states_)};
        HandleFree{}(error);
      }
    }

    //--------------------------------------------------------------------------
    /// \fn initialize
    /// \brief __global__ function wrapper for curand_init
    //--------------------------------------------------------------------------
    void initialize()
    {
      Details::setup_kernel<StateType, L><<<default_M_x_, default_N_x_>>>(
        raw_states_);      
    }

    //--------------------------------------------------------------------------
    /// \fn initialize
    /// \brief __device__ function wrapper for curand_init
    //--------------------------------------------------------------------------
    __device__ void initialize(
      const std::size_t tid,
      unsigned long long offset = 0)
    {
      curand_init(seed_, tid, offset, &raw_states_[tid]);
    }

    //--------------------------------------------------------------------------
    /// \brief Accessor to underlying CUDA C-style array
    //--------------------------------------------------------------------------
    StateType* raw_states()
    {
      return raw_states_;
    }


  private:

    // Default number of threads in a single thread block.
    const std::size_t default_N_x_;

    // Default number of thread blocks in a grid.
    const std::size_t default_M_x_;

    unsigned long long seed_;

    StateType* raw_states_; 

}; // class RawStates


//template <std::size_t L, typename StateType>
//RawStates<L, StateType>::RawStates(const std::size_t default_N_x):
//  default_N_x_{default_N_x},
//  default_M_x_{(L + default_N_x_ - 1)/ default_N_x_}
//{
//  Details::setup_kernel<StateType, L><<<default_M_x_, default_N_x_>>>(
//    raw_states_);
//}

} // namespace Curand

#endif // _CURAND_INITIAL_STATES_H_
