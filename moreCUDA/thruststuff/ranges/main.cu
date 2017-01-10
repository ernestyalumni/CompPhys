/** 
 * main.cu
 * 
 * \file main.cu
 * \author Ernest Yeung
 * \brief demonstrates examples of using various ranges (for thrust), including repeated_range
 * 
 * typed up by Ernest Yeung  ernestyalumni@gmail.com
 * \date 20170109
 * cf. https://github.com/thrust/thrust/blob/master/examples/repeated_range.cu
 * 
 * 
 * Compilation tip
 * nvcc -std=c++11 main.cu ./ranges/ranges.cu -o main.exe
 * 
 */
#include "./ranges/ranges.h"
#include <thrust/fill.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <iostream>

int main(void)
{
	thrust::device_vector<int> data(4);
	data[0] = 10;
	data[1] = 20;
	data[2] = 30;
	data[3] = 40;
	
	// print the initial data
	std::cout << "range			";
	thrust::copy(data.begin(), data.end(), std::ostream_iterator<int>(std::cout, " "));  	std::cout << std::endl;
	
	using Iterator = thrust::device_vector<int>::iterator ;
	
	// create repeated_range with elements repeated twice
	repeated_range<Iterator> twice(data.begin(), data.end(), 2);
	std::cout << "repeated x2:  ";
	thrust::copy(twice.begin(), twice.end(), std::ostream_iterator<int>(std::cout, " ")); std::cout << std::endl;
	
	// create repeated_range with elements repeated x3
	repeated_range<Iterator> thrice(data.begin(), data.end(), 3);
	std::cout << "repeated x3:  ";
	thrust::copy(thrice.begin(), thrice.end(), std::ostream_iterator<int>(std::cout, " ")); std::cout << std::endl;
	
	
	// Tiled range examples
	
	// create tiled_range with 2 tiles
	tiled_range<Iterator> two(data.begin(), data.end(), 2);
	std::cout << "two tiles:   ";
	thrust::copy(two.begin(), two.end(), std::ostream_iterator<int>(std::cout, " ")); std::cout << std::endl;
	
	// create tiled_range with 3 tiles
	tiled_range<Iterator> three(data.begin(), data.end(), 3);
	std::cout << "three tiles: ";
	thrust::copy(three.begin(), three.end(), std::ostream_iterator<int>(std::cout, " ")); std::cout << std::endl;
	
	
	
	return 0;
	
	
}
