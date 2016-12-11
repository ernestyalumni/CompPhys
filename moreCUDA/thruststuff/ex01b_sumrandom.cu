/* ex01b_sumrandom.cu
 * typed up by Ernest Yeung  ernestyalumni@gmail.com
 * 20160808
 * cf. https://thrust.github.io/
 * */
 /* Compilation note: for some reason, this works:
 * nvcc ex00_randomgen.cu -o ex00_randomgen.exe
 * but for 
 * nvcc -std=c++11 ex00_randomgen.cu -o ex00_randomgen.exe
 * I obtain these 2 errors: 
 * error: identifier "__builtin_ia32_monitorx" is undefined
 * and same for identifier "__builtin_ia32_mwaitx"
 * 
 * BUT when you include this flag: -D_MWAITXINTRIN_H_INCLUDED
 * it works, i.e.
 * nvcc -std=c++11 -D_MWAITXINTRIN_H_INCLUDED ex01b_sumrandom.cu -o ex01b_sumrandom.exe
 * */
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <algorithm>
#include <cstdlib>
#include <iostream>

int main(void)
{
	// generate random data serially
	thrust::host_vector<int> h_vec(100);
	std::generate(h_vec.begin(), h_vec.end(), rand);

	std::cout << " These are the first 10 values of the randomly generated h_vec " << std::endl;
	for (int i = 0; i<10; ++i) {
		std::cout << " The ith value, i : " << i << " value : " << h_vec[i] << std::endl; 
	}

	// transfer to device and compute sum
	thrust::device_vector<int> d_vec = h_vec;
	int x = thrust::reduce(d_vec.begin(), d_vec.end(), 0, thrust::plus<int>());

	std::cout << " This is the result of the summation, x : " << x << std::endl; 
	return 0;
	
}
 
