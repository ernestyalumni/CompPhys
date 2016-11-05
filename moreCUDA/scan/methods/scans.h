/* scans.h
 * Ernest Yeung
 * ernestyalumni@gmail.com
 * Demonstrates Hillis/Steele and Blelloch (exclusive) scan with a parallel implementation
 * with CUDA C/C++ and global memory
 * 
 * */
#ifndef __SCANS_H__
#define __SCANS_H__

#include <vector>
#include "checkerror.h" // checkCudaErrors

// parallel implementations

__global__ void Blelloch_up_global(float* f_in, float* f_out, const int k, const int L_x); 

__global__ void Blelloch_down_global( float* f_in, float* f_out, const int k, const int L_x); 

__global__ void copy_swap(float* f_in, float* f_target, const int L_x) ;

void Blelloch_scan_kernelLauncher(float* dev_f_in, float* dev_f_out, const int L_x, const int M_in); 

__global__ void HillisSteele_global(float* f_in, float* f_out, const int k, const int L_x); 

void HillisSteele_kernelLauncher(float* dev_f_in, float* dev_f_out, const int L_x, const int M_in) ;

// serial implementation

void blelloch_up( std::vector<float> f_in, std::vector<float> &f_out, const int k ) ;

void blelloch_down( std::vector<float> f_in, std::vector<float> &f_out, const int k); 

//void blelloch_serial( std::vector<float> &f_in, std::vector<float> &f_out, const int N);

void blelloch_serial( std::vector<float>& f_in ) ;

void HillisSteele_serial( std::vector<float>& f_in ) ;



#endif // __SCANS_H__
