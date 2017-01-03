/** sum.cu
 * 
 * \file sum.cu
 * \author Ernest Yeung
 * typed up by Ernest Yeung  ernestyalumni@gmail.com
 * \date 20170102
 * cf. http://istar.cse.cuhk.edu.hk/icuda/
 * icuda hands-on introduction to CUDA programming
 * 
 * Compilation tip
 * nvcc -std=c++11 sum.cu -o sum.exe
 * 
 * What this is really is mathematically is
 * \sum_{i=0}^{N-1} v[i], \forall \, v[i] \in K, K = \mathbb{Z} or \mathbb{R}, v \in K^N
 * 
 * */

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h> // thrust::plus< T > ; remember also minus,multiplies,divides,modulus,negate
#include <thrust/random.h>

int my_rand(void)
{
	static thrust::default_random_engine rng;
	static thrust::uniform_int_distribution<int> dist(0, 9999);
	return dist(rng);
}

int main(int argc, char* argv[]) 
{
	// Generate random data on the host.
	thrust::host_vector<int> h(100);
	thrust::generate(h.begin(), h.end(), my_rand ) ; 
	
	// Copy data from host to device.
	thrust::device_vector<int> d = h;
	
	// Compute sum on the device.
	int sum = thrust::reduce
	(
		d.begin(), d.end(), 	// Data range.
		0,						// Initial value of the reduction
		thrust::plus<int>()		// Binary operation used to reduce values.
	);
	
	// Print the sum.
	std::cout << sum << std::endl;
}
