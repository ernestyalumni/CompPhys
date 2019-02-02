//------------------------------------------------------------------------------
/// \file ErrorHandling.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Error Handling functions main driver file
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
///  nvcc --std=c++14 ErrorHandling_main.cpp ErrorHandling.cpp -o ErrorHandling_main
//------------------------------------------------------------------------------
#include "ErrorHandling.h"

#include <cuda_runtime_api.h> 
#include <iostream>
#include <stdexcept>

using Utilities::ErrorHandling::CUDA::HandleMalloc;

int main()
{
  // HandleMallocDefaultConstructs
  {
    std::cout << "\n HandleMallocDefaultConstructs \n";

    HandleMalloc handle_malloc;
  }

  // HandleMallocConstructsWithcudaSuccess
  {
    std::cout << "\n HandleMallocConstructsWithcudaSuccess \n";

    HandleMalloc handle_malloc {cudaSuccess};
  }

  // HandleMallocConstructsWithcudaErrorInvalidValue
  {
    std::cout << "\n HandleMallocConstructsWithcudaErrorInvalidValue \n";

    try
    {
      HandleMalloc handle_malloc {cudaErrorInvalidValue};
    }
    catch (const std::runtime_error& e)
    {
      std::cout << " Successfully caught this error: " << e.what() << '\n';
    }
  }

  // HandleMallocConstructsWithcudaErrorMemoryAllocation
  {
    std::cout << "\n HandleMallocConstructsWithcudaErrorMemoryAllocation \n";

    try
    {
      HandleMalloc handle_malloc {cudaErrorMemoryAllocation};
    }
    catch (const std::runtime_error& e)
    {
      std::cout << " Successfully caught this error: " << e.what() << '\n';
    }
  }

  // HandleMallocFunctionCallOperatorWorksWithcudaSuccess
  {
    std::cout << "\n HandleMallocFunctionCallOperatorWorksWithcudaSuccess \n";

    HandleMalloc{}(cudaSuccess);
  }

  // HandleMallocFunctionCallOperatorWorksWithcudaErrorInvalidValue
  {
    std::cout <<
      "\n HandleMallocFunctionCallOperatorWorksWithcudaErrorInvalidValue \n";

    try
    {
      HandleMalloc{}(cudaErrorInvalidValue);
    }
    catch (const std::runtime_error& e)
    {
      std::cout << " Successfully caught this error: " << e.what() << '\n';
    }
  }

  // HandleMallocFunctionCallOperatorWorksWithcudaErrorMemoryAllocation
  {
    std::cout <<
      "\n HandleMallocFunctionCallOperatorWorksWithcudaErrorMemoryAllocation \n";

    try
    {
      HandleMalloc{}(cudaErrorMemoryAllocation);
    }
    catch (const std::runtime_error& e)
    {
      std::cout << " Successfully caught this error: " << e.what() << '\n';
    }
  }
}