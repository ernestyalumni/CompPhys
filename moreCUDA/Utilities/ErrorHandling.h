//------------------------------------------------------------------------------
/// \file ErrorHandling.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Error Handling functions.
/// \url https://docs.nvidia.com/cuda/cuda-runtime-api/
/// \ref 
/// \details Error handling function wrappers of CUDA runtime application
/// programming interface.
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
///  nvcc --std=c++14 ErrorHandling_main.cpp ErrorHandling.cpp -o
///   ErrorHandling_main
//------------------------------------------------------------------------------
#ifndef _UTILITIES_ERROR_HANDLING_H_
#define _UTILITIES_ERROR_HANDLING_H_

#include <cuda_runtime_api.h> // cudaError_t
#include <string>

namespace Utilities
{

namespace ErrorHandling
{

namespace CUDA
{


//------------------------------------------------------------------------------
/// \class HandleError
/// \brief Virtual base class for C++ functor for checking the result of a
/// CUDA function call.
/// Typically, it deals with the return value of cudaMalloc:
/// __host__ __device__ cudaError_t cudaMalloc(void** devPtr, size_t size)
/// and which can return 3 things:
/// cudaSuccess, cudaErrorInvalidValue, cudaErrorMemoryAllocation
//------------------------------------------------------------------------------
class HandleError
{
  public:

    virtual void operator()(const cudaError_t error) = 0;

  protected:

    // Accessors

    virtual cudaError_t error() const = 0;

    virtual std::string error_name() const = 0;

    virtual std::string error_string() const = 0;


    //--------------------------------------------------------------------------
    /// \fn peek_at_last_error
    /// \brief Returns the last error produced by any runtime calls in same host
    /// thread.
    /// \details This call doesn't reset error to cudaSuccess like
    /// cudaGetLastError()
    //--------------------------------------------------------------------------
    virtual cudaError_t peek_at_last_error() const
    {
      return cudaPeekAtLastError();
    }
};

class HandleMalloc : public HandleError
{
  public:

    HandleMalloc();

    HandleMalloc(const cudaError_t error);

    void operator()(const cudaError_t error);

  protected:

    cudaError_t error() const
    {
      return error_;
    }

    std::string error_name() const
    {
      return std::string{cudaGetErrorName(error_)};
    }

    std::string error_string() const
    {
      return std::string{cudaGetErrorString(error_)};
    }

  private:

    cudaError_t error_;
};

} // namespace CUDA

} // namespace ErrorHandling

} // namespace Utilities

#endif // _UTILITIES_ERROR_HANDLING_CUDA_H_
