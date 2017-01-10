/** 
 * ranges.h
 * 
 * \file ranges.h
 * \author Ernest Yeung
 * \brief various ranges (for thrust), including repeated_range
 * 
 * typed up by Ernest Yeung  ernestyalumni@gmail.com
 * \date 20170103
 * cf. https://github.com/thrust/thrust/blob/master/examples/repeated_range.cu
 * 
 * 
 * Compilation tip
 * nvcc -std=c++11 repeated_range.cu -o repeated_range.exe
 * 
 */
#ifndef __RANGES_H__
#define __RANGES_H__

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/functional.h> // unary_function
//#include <thrust/fill.h>


// this example illustrates how to make repeated access to a range of values
// examples:
//   repeated_range([0, 1, 2, 3], 1) -> [0, 1, 2, 3]
//   repeated_range([0, 1, 2, 3], 2) -> [0, 0, 1, 1, 2, 2, 3, 3]
//   repeated_range([0, 1, 2, 3], 3) -> [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
//

template <typename Iterator>
class repeated_range
{
	public:
	
	using difference_type = typename thrust::iterator_difference<Iterator>::type ; 

	struct repeat_functor : public thrust::unary_function<difference_type,difference_type>
	{
		difference_type repeats;
		
		repeat_functor(difference_type repeats)
			: repeats(repeats) {}
			
			
		__host__ __device__ 
		difference_type operator()(const difference_type& i) const 
		{
				return i / repeats;
		}
	};
	
	using CountingIterator    = typename thrust::counting_iterator<difference_type> ;
	using TransformIterator = typename thrust::transform_iterator<repeat_functor, CountingIterator> ;
	using PermutationIterator = typename thrust::permutation_iterator<Iterator, TransformIterator> ;
	
	// type of the repeated_range iterator
	using iterator = PermutationIterator; 
	 
	// construct repeated_range for the range [first,last)
	repeated_range(Iterator first, Iterator last, difference_type repeats) 
			: first(first), last(last), repeats(repeats) {}

		
	iterator begin(void) const 
	{
		return PermutationIterator(first, TransformIterator(CountingIterator(0), repeat_functor(repeats)));
	}
	
	iterator end(void) const 
	{
		return begin() + repeats * (last - first);
	}

	
	protected: 
	Iterator first;
	Iterator last;
	difference_type repeats;
};

// this example illustrates how to tile a range multiple times
// examples:
//   tiled_range([0, 1, 2, 3], 1) -> [0, 1, 2, 3]
//   tiled_range([0, 1, 2, 3], 2) -> [0, 1, 2, 3, 0, 1, 2, 3]
//   tiled_range([0, 1, 2, 3], 3) -> [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
//

template <typename Iterator>
class tiled_range
{
	public:
	
	using difference_type = typename thrust::iterator_difference<Iterator>::type ;
	
	struct tile_functor : public thrust::unary_function<difference_type,difference_type>
	{
		difference_type tile_size;
		
		tile_functor(difference_type tile_size)
			: tile_size(tile_size) {}
			
		__host__ __device__
		difference_type operator()(const difference_type &i) const
		{
			return i % tile_size;
		}
	};
	
	using CountingIterator  = typename thrust::counting_iterator<difference_type> ;
	using TransformIterator = typename thrust::transform_iterator<tile_functor, CountingIterator> ;
	using PermutationIterator = typename thrust::permutation_iterator<Iterator,TransformIterator> ;
	
	// type of the tiled_range iterator
	using iterator = PermutationIterator ;
	
	// construct repeated_range for the range [first,last)
	tiled_range(Iterator first, Iterator last, difference_type tiles)
		: first(first), last(last), tiles(tiles) {}
		
	iterator begin(void) const
	{
		return PermutationIterator(first, TransformIterator(CountingIterator(0), tile_functor(last - first)));
	}
	
	iterator end(void) const
	{
		return begin() + tiles * (last - first);
	}
	
	protected:
	Iterator first;
	Iterator last;
	difference_type tiles;
};
	


#endif // __RANGES_H__


/* Further explanations (for pedagogical, learning purposes) of what things mean
 * 
 * cf. http://blog.ethanlim.net/2014/07/separate-c-template-headers-h-and.html
 * Separate C++ Template Headers (*.h) and Implementation files (*.cpp) 
 * very clear explanation of the problem with explicit instantiation
 * 
 * 
 * thrust::unary_function<difference_type,difference_type>
 * so we need #include <thrust/functional.h> 
 * 
 * No documentation or examples on thrust page, only the header file code:
 * cf. https://thrust.github.io/doc/functional_8h_source.html
 * 0 struct unary_function
   71 {
   75   typedef Argument argument_type;
   76 
   80   typedef Result   result_type;
   81 }; // end unary_function
   * so it's just a struct
   * 
   * */
