/** maxSquaredDiff.cu
 * 
 * \file maxSquaredDiff.cu
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
 * a,b \in K^N
 * \max_{i \in \lbrace 0,\dots , N-1}{ (a[i]-b[i])^2 }
 * where K in general is a field
 * 
 * */
#include <iostream>
#include <thrust/inner_product.h>  
#include <thrust/functional.h>  // thrust::maximum< T > , generalized Identity operations, it is an Adaptable Binary Function
#include <thrust/device_vector.h>

using namespace thrust::placeholders; // _1, _2

// cf. https://thrust.github.io/doc/group__transformed__reductions.html#ga321192d85c5f510e52300ae762c7e995
/*
 * Note the first 2 forms of thrust::inerproduct
 * template<typename DerivedPolicy, typename InputIterator1, typename input ..
 * what's important are the Parameters
 * 
 * exec  The execution policy to use for parallelization
 * first1 The beginning of the first sequence
 * last1 The end of the first sequence
 * first2 The beginning of the second sequence.
 * init   Initial value of the result
 * 
 * sum init + (*first1 * *first2) + (*(first1+1) * *(first2+1))
 * 
 * another version (overloaded) has parameters
 * first1
 * last1
 * first2
 * init
 * 
 * and computes sum init + (*first1 * *first2) + (*(first1+1) * *(first2+1)) + ...
 * 
 * and then a generalized version of inner product
 * Parameters
 * exec
 * first1
 * last1
 * first2
 * init
 * binary_op1
 * binary_op2
 * 
 * computes sum binary_op1( init, binary_op2(*first1, *first2)), 
 * 
 * mathematically, given binary operations +,*, for this ring,
 * \bigoplus_{i=0}^{N-1} ( a[i] * b[i] )
 * a,b \in K^N, where b \in K^{N_1} and N \leq N_1
 * but that it's more general:
 * given binary operations +, g, where g: K\times K \to K, 
 * g(a[i], b[i]) \in K, 
 * this computes
 * \bigoplus_{i=0}^{N-1} g(a[i],b[i])
 * 
 * also another version has 
 * 
 * first1
 * last1
 * first2
 * init
 * binary_op1
 * binary_op2
 * 
 * */

int main(int argc, char* argv[])
{
	// Initialize 2 vectors.
	thrust::device_vector<float> a(4);
	thrust::device_vector<float> b(4);
	a[0] = 1.0; b[0] = 2.0;
	a[1] = 2.0; b[1] = 4.0;
	a[2] = 3.0; b[2] = 3.0;
	a[3] = 4.0; b[3] = 0.0;
	
	// Compute the maximum squared difference.
	float max_squared_diff = thrust::inner_product 
	(
		a.begin(), a.end(), 		// Data range 1.
		b.begin(), 					// Data range 2.
		0,							// Initial value for the reduction.
		thrust::maximum<float>(), 	// Binary operation used to reduce values.
		(_1 - _2) * (_1 - _2) 		// Lambda expression to compute squared difference.
	);
	
	
	// Print the result.
	std::cout << max_squared_diff << std::endl;
}

