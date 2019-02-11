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
#include <type_traits>

namespace Curand
{

template <std::size_t L, typename StateType = curandState_t>
class Uniform;

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

//--------------------------------------------------------------------------
/// \fn initialize_and_generate_uniform_kernel
/// \brief Function wrapper for curand_init and curand_uniform
//--------------------------------------------------------------------------
template <typename StateType, std::size_t L>
__global__ void initialize_and_generate_uniform_kernel(
  RawStates<L, StateType>& raw_states,
  Uniform<L, StateType>& uniform)
{
  // Get our global thread ID.
  std::size_t k_x {blockIdx.x * blockDim.x + threadIdx.x};

  // Make sure we don't go out of bounds.
  if (k_x >= L) { return; }  

  for (std::size_t tid {k_x}; tid < L; tid += gridDim.x * blockDim.x)
  {
//    raw_states.initialize(tid, 0);

//    uniform.generate(tid);
  }
}

template <typename StateType, std::size_t L>
__global__ void initialize_and_generate_uniform_kernel(
  Initialize<L, StateType>& initialize,
  float* sequence)
{
  // Get our global thread ID.
  std::size_t k_x {blockIdx.x * blockDim.x + threadIdx.x};

  // Make sure we don't go out of bounds.
  if (k_x >= L) { return; }  

  for (std::size_t tid {k_x}; tid < L; tid += gridDim.x * blockDim.x)
  {
//    initialize()(tid, 0);
    initialize.do_initialize(tid);
//    initialize->do_initialize(tid, 0);

    // Copy state to local memory for efficiency
//    StateType local_state = (initialize.states())[tid]; 
//    StateType local_state = (initialize->states())[tid]; 

    // Generate pseudo-random uniform, and store results
  //  sequence[tid] = curand_uniform(&local_state);

    // Copy state back to global memory
    //(initialize.states())[tid] = local_state;
    //(initialize->states())[tid] = local_state;
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
//template <std::size_t L, typename StateType = curandState_t>
template <std::size_t L, typename StateType>
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

    __device__ void generate(const std::size_t tid)
    {
      float x;

      // Copy state to local memory for efficiency
      StateType local_state = (raw_states_.raw_states())[tid]; 

      // Generate pseudo-random uniform
      x = curand_uniform(&local_state);

      // Copy state back to global memory
      (raw_states_.raw_states())[tid] = local_state;

      // Store results
      dev_sequence_[tid] = x;      
    }

//    __device__ void operator()(const std::size_t tid)
    __device__ void initialize_and_generate(const std::size_t tid)
    //  typename std::enable_if_t<!is_initialized_, const std::size_t> tid)
    {
      raw_states_.initialize(tid);

      generate(tid);
    }

    //__device__ void operator()(
    //  typename std::enable_if_t<is_initialized_, const std::size_t> tid)
    //{
    //  generate(tid);
    //}

    //--------------------------------------------------------------------------
    /// \fn operator()()
    /// \brief Function call operator overload to generate uniform pseudo-random
    /// floating point numbers from 0.0 to 1.0, i.e. (0.0, 1.0]
    //--------------------------------------------------------------------------
    void operator()()
    {
      if (is_initialized_)
      {
        Details::generate_uniform_kernel<
          StateType,
          L>
          <<<default_M_x_, default_N_x_>>>(
            raw_states_.raw_states(), dev_sequence_);
      }
      else
      {
        Details::initialize_and_generate_uniform_kernel<
          StateType,
          L>
          <<<default_M_x_, default_N_x_>>>(raw_states_, *this);       

        is_initialized_ = true;
      }
    }

    // Copy device memory to host
    void copy_to_host(float* h_data) const
    {
      const cudaError_t error {
        cudaMemcpy(
          h_data, dev_sequence_, L * sizeof(float), cudaMemcpyDeviceToHost)};
      
      HandleMemcpy{}(error);
    }

    // Accessors

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

    bool is_initialized_;

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
  raw_states_{default_N_x, seed},
  is_initialized_{false}
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

  is_initialized_ = true;
}

template <std::size_t L, typename StateType>
Uniform<L, StateType>::Uniform(const std::size_t default_N_x):
  Uniform{default_N_x, 1234}
{}

//------------------------------------------------------------------------------
/// \class GenerateUniform
/// \details Provide a wrapper to curand_uniform.
/// \tparam State Parameter is meant to be 
//------------------------------------------------------------------------------
template <
  typename StateType = curandState_t//,
//  typename std::enable_if_t<
  //    (std::is_same<StateType, curandState_t>::value ||
    //    std::is_same<StateType, curandStatePhilox4_32_10_t>::value ||
      //    std::is_same<StateType, curandStateMRG32k3a>::value ||
        //    std::is_same<StateType, curandState>::value)>* = nullptr
  >
class GenerateUniform
{
  public:

    GenerateUniform() = delete;

    // Not movable, not copyable.
    GenerateUniform(const GenerateUniform&) = delete; // copy ctor
    GenerateUniform(GenerateUniform&&) = delete; // move ctor
    // copy assignment ctor
    GenerateUniform& operator=(const GenerateUniform&) = delete;
    // move assignment ctor
    GenerateUniform& operator=(GenerateUniform&&) = delete;

    ~GenerateUniform() = default;

    __device__ void operator()(InitialState<StateType>& initial_state)
    {
      // Copy state to local memory for efficiency
      StateType local_state = initial_state.state();

      // Generate pseudo-random uniform
      float x {curand_uniform(&local_state)};

      // Copy state back to global memory
      initial_state.set_state(local_state);

      // Store results
      x_ = x;      
    }

    //--------------------------------------------------------------------------
    /// \brief Accessor to underlying float
    //--------------------------------------------------------------------------

    __device__ float& x()
    {
      return x_;
    }

  private:

    float x_;
}; // GenerateUniform


//--------------------------------------------------------------------------
/// \fn generate_uniform_kernel
/// \brief Function wrapper for curand_uniform
//--------------------------------------------------------------------------
template <typename StateType, std::size_t L>
__global__ void generate_uniform_kernel(
  InitialState<StateType>* states,
  GenerateUniform<StateType>* sequence)
{
  // Get our global thread ID.
  std::size_t k_x {blockIdx.x * blockDim.x + threadIdx.x};

  // Make sure we don't go out of bounds.
  if (k_x >= L) { return; }  

  for (std::size_t tid {k_x}; tid < L; tid += gridDim.x * blockDim.x)
  {
    (sequence[tid])()(states[tid]);
  }
}

template <typename StateType, std::size_t L>
__global__ void initialize_and_generate_uniform_kernel(
  InitialState<StateType>* states,
  GenerateUniform<StateType>* sequence,
  const unsigned long long seed = 1234)
{
  // Get our global thread ID.
  std::size_t k_x {blockIdx.x * blockDim.x + threadIdx.x};

  // Make sure we don't go out of bounds.
  if (k_x >= L) { return; }  

  for (std::size_t tid {k_x}; tid < L; tid += gridDim.x * blockDim.x)
  {
    (states[tid])(seed, tid);
//    (sequence[tid])(states[tid]);
  }
}

//------------------------------------------------------------------------------
/// \class UniformDistribution
/// \details Provide a wrapper to curand_uniform.
/// \tparam State Parameter is meant to be 
//------------------------------------------------------------------------------
template <
  std::size_t L,
  typename StateType = curandState_t
  >
class UniformDistribution
{
  public:

    UniformDistribution() = delete;

    //--------------------------------------------------------------------------
    /// \param default_N_x Default number of threads in a single block.
    //--------------------------------------------------------------------------
    UniformDistribution(const std::size_t N_x, const unsigned long long seed);

    explicit UniformDistribution(const std::size_t N_x);

    ~UniformDistribution();

    //--------------------------------------------------------------------------
    /// \brief Accessors to underlying CUDA C-style array
    //--------------------------------------------------------------------------

    InitialState<StateType>& initial_states()
    {
      return initial_states_;
    }

    GenerateUniform<StateType>& sequence()
    {
      return sequence_;
    }

  private:

    using HandleFree = Utilities::ErrorHandling::CUDA::HandleFree;
    using HandleMalloc = Utilities::ErrorHandling::CUDA::HandleMalloc;
    using HandleMemcpy = Utilities::ErrorHandling::CUDA::HandleMemcpy;
    using HandleMemset = Utilities::ErrorHandling::CUDA::HandleMemset;

    // Default number of threads in a single thread block.
    const std::size_t N_x_;

    // Default number of thread blocks in a grid.
    const std::size_t M_x_;

    InitialState<StateType>* initial_states_;

    bool is_initialized_;

    GenerateUniform<StateType>* sequence_;
}; // UniformDistribution

template <std::size_t L, typename StateType>
UniformDistribution<L, StateType>::UniformDistribution(
  const std::size_t N_x,
  const unsigned long long seed
  ):
  N_x_{N_x},
  M_x_{(L + N_x_ - 1) / N_x_},
  is_initialized_{false}
{
  // Allocate space for results on device.
  cudaError_t error {
    cudaMalloc((void**)&sequence_, L * sizeof(GenerateUniform<StateType>))};
  HandleMalloc{}(error);

  // Set results to 0
  error = cudaMemset(sequence_, 0, L * sizeof(GenerateUniform<StateType>));
  HandleMalloc{}(error);

  // Generate and use uniform pseudo-random

  initialize_and_generate_uniform_kernel<StateType, L><<<M_x_, N_x_>>>(
    initial_states_,
    sequence_,
    seed);
}

template <std::size_t L, typename StateType>
UniformDistribution<L, StateType>::UniformDistribution(const std::size_t N_x):
  UniformDistribution{N_x, 1234}
{}

template <std::size_t L, typename StateType>
UniformDistribution<L, StateType>::~UniformDistribution()
{
  if (sequence_ != nullptr)
  {
    const cudaError_t error {cudaFree(sequence_)};
    HandleFree{}(error);
  }

  if (initial_states_ != nullptr)
  {
    const cudaError_t error {cudaFree(initial_states_)};
    HandleFree{}(error);
  }
}


#if 0
template <
  std::size_t L,
  typename StateType = curandState_t,
  typename std::enable_if_t<
      (std::is_same<StateType, curandState_t>::value ||
        std::is_same<StateType, curandStatePhilox4_32_10_t>::value ||
          std::is_same<StateType, curandStateMRG32k3a>::value ||
            std::is_same<StateType, curandState>::value)>* = nullptr
  >
class GenerateUniform
{
  public:

    GenerateUniform() = delete;

    //--------------------------------------------------------------------------
    /// \param default_N_x Default number of threads in a single block.
    //--------------------------------------------------------------------------
    GenerateUniform(const std::size_t N_x, const unsigned long long seed):
      N_x_{N_x},
      M_x_{(L + N_x_ - 1)/ N_x_},
      initialize_{N_x, seed},
      is_initialized_{false}
    {
      // Allocate space for results on device.
      cudaError_t error {cudaMalloc((void**)&sequence_, L * sizeof(float))};
      HandleMalloc{}(error);

      // Set results to 0
      error = cudaMemset(sequence_, 0, L * sizeof(float));
      HandleMalloc{}(error);
    }

    explicit GenerateUniform(const std::size_t N_x):
      GenerateUniform{N_x, 1234}
    {}

    ~GenerateUniform()
    {
      if (sequence_ != nullptr)
      {
        const cudaError_t error {cudaFree(sequence_)};
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
      if (is_initialized_)
      {
        Details::generate_uniform_kernel<
          StateType,
          L>
          <<<M_x_, N_x_>>>(initialize_.states(), sequence_);
      }
      else
      {
        Details::initialize_and_generate_uniform_kernel<
          StateType,
          L>
          <<<M_x_, N_x_>>>(initialize_, sequence_);

        is_initialized_ = true;
      }
    }

    // Copy device memory to host
    void copy_to_host(float* h_data) const
    {
      const cudaError_t error {
        cudaMemcpy(
          h_data, sequence_, L * sizeof(float), cudaMemcpyDeviceToHost)};
      
      HandleMemcpy{}(error);
    }

    // Accessors

    float* sequence()
    {
      return sequence_;
    }

  private:

    using HandleFree = Utilities::ErrorHandling::CUDA::HandleFree;
    using HandleMalloc = Utilities::ErrorHandling::CUDA::HandleMalloc;
    using HandleMemcpy = Utilities::ErrorHandling::CUDA::HandleMemcpy;
    using HandleMemset = Utilities::ErrorHandling::CUDA::HandleMemset;

    // Default number of threads in a single thread block.
    const std::size_t N_x_;

    // Default number of thread blocks in a grid.
    const std::size_t M_x_;

    Initialize<L, StateType> initialize_;

    bool is_initialized_;

    float* sequence_;
}; // class GenerateUniform
#endif 

#if 0
template <
  std::size_t L,
  typename StateType = curandState_t,
  typename std::enable_if_t<
      (std::is_same<StateType, curandState_t>::value ||
        std::is_same<StateType, curandStatePhilox4_32_10_t>::value ||
          std::is_same<StateType, curandStateMRG32k3a>::value ||
            std::is_same<StateType, curandState>::value)>* = nullptr
  >
class GenerateUniform
{
  public:

    GenerateUniform() = delete;

    //--------------------------------------------------------------------------
    /// \param default_N_x Default number of threads in a single block.
    //--------------------------------------------------------------------------
    GenerateUniform(const std::size_t N_x, const unsigned long long seed):
      N_x_{N_x},
      M_x_{(L + N_x_ - 1)/ N_x_},
      initialize_{N_x, seed},
      is_initialized_{false}
    {
      // Allocate space for results on device.
      cudaError_t error {cudaMalloc((void**)&sequence_, L * sizeof(float))};
      HandleMalloc{}(error);

      // Set results to 0
      error = cudaMemset(sequence_, 0, L * sizeof(float));
      HandleMalloc{}(error);
    }

    explicit GenerateUniform(const std::size_t N_x):
      GenerateUniform{N_x, 1234}
    {}

    ~GenerateUniform()
    {
      if (sequence_ != nullptr)
      {
        const cudaError_t error {cudaFree(sequence_)};
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
      if (is_initialized_)
      {
        Details::generate_uniform_kernel<
          StateType,
          L>
          <<<M_x_, N_x_>>>(initialize_.states(), sequence_);
      }
      else
      {
        Details::initialize_and_generate_uniform_kernel<
          StateType,
          L>
          <<<M_x_, N_x_>>>(initialize_, sequence_);

        is_initialized_ = true;
      }
    }

    // Copy device memory to host
    void copy_to_host(float* h_data) const
    {
      const cudaError_t error {
        cudaMemcpy(
          h_data, sequence_, L * sizeof(float), cudaMemcpyDeviceToHost)};
      
      HandleMemcpy{}(error);
    }

    // Accessors

    float* sequence()
    {
      return sequence_;
    }

  private:

    using HandleFree = Utilities::ErrorHandling::CUDA::HandleFree;
    using HandleMalloc = Utilities::ErrorHandling::CUDA::HandleMalloc;
    using HandleMemcpy = Utilities::ErrorHandling::CUDA::HandleMemcpy;
    using HandleMemset = Utilities::ErrorHandling::CUDA::HandleMemset;

    // Default number of threads in a single thread block.
    const std::size_t N_x_;

    // Default number of thread blocks in a grid.
    const std::size_t M_x_;

    InitialState* initial_states_;

    bool is_initialized_;

    float* sequence_;
}; // class GenerateUniform
#endif 

} // namespace Curand

#endif // _CURAND_UNIFORM_H_