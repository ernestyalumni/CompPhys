"""
@file RangeFilter.py
@author Ernest Yeung, <ernestyalumni@gmail.com>  

@note make "boilerplate" test values with np.random.uniform, 
		e.g. np.random.uniform(200,500,4)
"""
import numpy
import numpy as np 

def rangeFilter(x,range_min,range_max): 
	"""
	@fn rangeFilter = rangeFilter(x,range_min,range_max)
	@param x : Numpy array, represents a scan, array of input values 
			range_min : minimum value for the desired range
			range_max : maximum value for the desired range 
	@returns x : Numpy array, representing the filtered values 
	"""
	x[x<range_min]=range_min
	x[x>range_max]=range_max
	return x

	
	
