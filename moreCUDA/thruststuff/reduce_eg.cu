/* reduce_eg.cu
 * demonstrates the use of reduce via thrust
 * \author Ernest Yeung  
 * \email ernestyalumni@gmail.com
 * \date 20161210
 * 
 * */
/* Compilation note: for some reason, this works:
 * nvcc -std=c++11 reduce_eg.cu -o reduce_eg.exe
 * without need for flag -D_MWAITXINTRIN_H_INCLUDED
 * */
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>

#include <thrust/sequence.h> // thrust::sequence

#include <iostream>


int main(int argc, char* argv[]) {
	// cf. http://docs.nvidia.com/cuda/thrust/#axzz4SUVUla3T  NVIDIA CUDA Toolkit Documentation, v8.0, Thrust
	
	// H has storage for 11 floats
	thrust::host_vector<float> H(11) ;
	
	// initialize individual elements
	for (auto iter = H.begin(); iter != H.end(); ++iter) {
		*iter = 1.0 ; }
	
	// H.size() returns the size of vector H
	std::cout << "H has size " << H.size() << std::endl;
	
	// print contents of H
	for (auto iter : H ) { std::cout << iter << " "; } 
	std::cout << std::endl; 
	
	// initialize all 11 floats of a device_vector to 1.0
	thrust::device_vector<float> D(11,1.0);
	
	// set the elements of D to 0, 1, 2, 3, ...
	thrust::sequence(D.begin(), D.end());
	
	// print D
	for (auto i = 0; i < D.size(); i++) { std::cout << "D[" << i << "] = " << D[i] << std::endl;  }
		
	// reduce
	float temp_val = 0.0;
	temp_val = thrust::reduce(D.begin(), D.end(), 0, thrust::plus<float>());
		
	std::cout << " This is the result of the summation of D : " << temp_val << std::endl; 
	
		
	return 0;
	
}

