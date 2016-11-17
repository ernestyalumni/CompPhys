/*
 * checkerror.h
 * 
 * EXAMPLE of USAGE:
 * checkCudaErrors( CUDA commandyouwanttocheckerrorsfor ) ;
 * 
 * */
#ifndef __ERRORS_H__
#define __ERRORS_H__

#include <iostream> // std::cerr, std::endl;
#include <cuda.h>
#include <cuda_runtime.h> // cudaSuccess
#include <cuda_runtime_api.h>

#define checkCudaErrors(val) checkerror( (val), #val, __FILE__, __LINE__ )

template<typename T>
void checkerror(T err, const char* const func, const char* const file, const int line) {
	if (err != cudaSuccess) {
		std::cerr << "CUDA error at: " << file << " : " << line << std::endl;
		std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
		exit(1);
	}
}

#endif // __ERRORS_H__
