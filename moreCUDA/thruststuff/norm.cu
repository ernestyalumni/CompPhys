/** norm.cu
 * 
 * \file norm.cu
 * \author Ernest Yeung
 * typed up by Ernest Yeung  ernestyalumni@gmail.com
 * \date 20170102
 * cf. http://istar.cse.cuhk.edu.hk/icuda/
 * icuda hands-on introduction to CUDA programming
 * 
 * Compilation tip
 * nvcc -std=c++11 norm.cu -o norm.exe
 * 
 * What this is really is mathematically is
 * S = \sum_{i=0}^{N-1} v[i]^2, \forall \, v[i] \in K, K = \mathbb{Z} or \mathbb{R}, v \in K^N
 * where K in general is a field
 * and then computing \sqrt{ S }
 * */
#include <cmath>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>  // thrust::plus< T > ; remember also minus,multiplies,divides,modulus,negate
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

// cf. https://thrust.github.io/doc/namespaceplaceholders.html
/* 
 * placeholders Namespace Reference
 * Facilities for constructing simple functions inline.
 * Objects in the thrust::placeholders namespace may be used to create 
 * simple arithmetic functions inline in an algorithm invocation.
 * Combining placeholders such as _1 and _2 with arithmetic operations such as + 
 * creates an unnamed function object which applies the operation to their arguments.
 * 
 * */
// note that this using namespace can be anywhere for desired scope
using namespace thrust::placeholders;  // _1, _2

int main(int argc, char* argv[])
{
	// Initialize host data.
	float h[4] = {1.0, 2.0, 3.0, 4.0};
	
	// Copy data from host to device.
	thrust::device_vector<float> d(h, h + 4);
	
	
	// Compute norm square.
	float norm2 = thrust::transform_reduce
	(
		d.begin(), d.end(),  	// Data range
		_1 * _1, 				// Unary transform operation.
		0,						// Initial value of the reduction.
		thrust::plus<float>()			// Binary operation used to reduce values.
	);
	
	// Compute norm.
	float norm = std::sqrt(norm2);
	
	// Print the norm.
	std::cout << norm << std::endl;
}

