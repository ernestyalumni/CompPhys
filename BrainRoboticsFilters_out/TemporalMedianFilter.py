"""
@file TemporalMedianFilter.py
@author Ernest Yeung, <ernestyalumni@gmail.com>  

@note make "boilerplate" test values with np.random.uniform, 
		e.g. np.random.uniform(200,500,(4,2))
"""
import numpy 
import numpy as np

def temporalMedianFilter(D,x_current,x_previous=None): 
	"""
	@fn temporalMedianFilter 
	@param D : positive integer, how many previous scans to keep
	@param x_current : Numpy array of size dimensions (1,N) (should be a row array)
	@param x_previous : Numpy array of size dimensions (D,N), D can be 0, 
						so that's why we have the None default value
	@returns y : Numpy array of size dimensions (1,N), with filtered values for the 
					median and 
			x_previous : D-1 rows of previous values, with the current received scan 
							x_current "at the bottom"
							
			Return these as a tuple (y,x_previous)				
	"""
	if (x_previous is None):
		return (x_current,x_current)
	else:
		D_previous, N_previous = x_previous.shape 
		if (D_previous < D):
			x_previous = np.vstack((x_previous,x_current))
			y = np.median(x_previous,axis=0)
			return (y,x_previous)
#		elif (D_previous >= D):
		else:
			x_previous = np.vstack((x_previous,x_current))
			x_previous = np.delete( x_previous, range(D_previous-D), axis=0)
			y = np.median(x_previous,axis=0)
			x_previous = np.delete(x_previous,(0),axis=0)
			return (y,x_previous)
			
