//------------------------------------------------------------------------------
/// \file HandleErrors.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief Error Handling functions.
/// \url https://docs.nvidia.com/cuda/cuda-runtime-api/
/// \ref 5.3. Error Handling, 5. Modules, CUDA Runtime API, CUDA Toolkit Doc. 
/// https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__ERROR.html
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
///    ErrorHandling_main
//------------------------------------------------------------------------------
#include "ErrorHandling.h"

#include <cuda_runtime_api.h> 
#include <stdexcept> // std::runtime_error

namespace Utilities
{

namespace ErrorHandling
{

namespace CUDA
{

HandleMalloc::HandleMalloc():
  error_{}
{}

HandleMalloc::HandleMalloc(const cudaError_t error):
  error_{error}
{
  HandleMalloc::operator()(error_);
}

void HandleMalloc::operator()(const cudaError_t error)
{
  error_ = error;

  if (error != cudaSuccess)
  {
    throw std::runtime_error(std::string{cudaGetErrorName(error)} + " " +
      std::string{cudaGetErrorString(error)});
  }
  return;
}

} // namespace CUDA

} // namespace ErrorHandling

} // namespace Utilities