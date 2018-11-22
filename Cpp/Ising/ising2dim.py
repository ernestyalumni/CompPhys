"""
@file ising2dim.py
@brief 2-dim. Ising model
@details cf. https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.item.html  
numpy.ndarray.item  
item is very similar to a[args], except, instead of an array scalar, a standard Python scalar is returned.   
This can be useful for speeding up access to elements of the array and 
doing arithmetic on elements of the array using Python's optimized math
"""
import numpy
import numpy as np
import sys
import math  

def periodic(i, limit, add):
	"""
	@fn periodic
	@brief Choose correct matrix index with periodic 
	boundary conditions  
	
	Input : 
	@param - i     : Base index  
	@param - limit : Highest \"legal\" index  
	@param - add   : Number to add or subtract from i 
	"""
	return (i + limit + add ) % limit


def monteCarlo(temp, size, trials):
	"""
	@fn monteCarlo
	@brief Calculate the energy and magnetization 
			(\"straight\" and squared) for a given temperature 
	
	Input:
	@param - temp : 	Temperature to calculate for 
	@param - size :		dimension of square matrix 
	@param - trials : 	Monte Carlo trials (how many times do we 
											flip the matrix?)  
	
	Output:  
	- E_av: 		Energy of matrix averaged over trials, normalized to spins**2  
	- E_variance :  Variance of energy, same normalization * temp**2
	- M_av:			Magnetic field of matrix, averaged over trials, normalized to spins**2 
	- M_variance : 	Variance of magnetic field, same normalization * temp 
	- Mabs : 		Absolute value of magnetic field, averaged over trials 
	
	@details cf. https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.item.html  
	numpy.ndarray.item  
	item is very similar to a[args], except, instead of an array scalar, a standard Python scalar is returned. 
	This can be useful for speeding up access to elements of the array and 
	doing arithmetic on elements of the array using Python's optimized math.
	"""
	
	# Setup spin matrix, initialize to ground state
	spin_matrix = np.zeros( (size,size), np.int8) + 1
	
	# Create and initialize variables 
	E 		= M 	=  0 
	E_av 	= E2_av = M_av = M2_av = Mabs_av = 0 
	
	# Setup array for possible energy changes 
	w = np.zeros(17,np.float64) 
	for de in xrange(-8,9,4): # include +8 
		w[de+8] = math.exp(-de/temp)  
		
	# Calculate initial magnetization: 
	M = spin_matrix.sum()
	# Calculate initial energy 
	for j in xrange(size):
		for i in xrange(size): 
			E -= spin_matrix.item(i,j) * \
					(spin_matrix.item(periodic(i,size,-1),j) + spin_matrix.item(i,periodic(j,size,1)))  
					
	# Start Metropolis Monte Carlo computation 
	for i in xrange(trials):
		# Metropolis
		# Loop over all spins, pick a random spin each time 
		for s in xrange(size**2): 
			x = int(np.random.random()*size) 
			y = int(np.random.random()*size) 
			deltaE = 2*spin_matrix.item(x,y) * \
						(spin_matrix.item(periodic(x,size,-1), y) + \
						spin_matrix.item(periodic(x,size,1), y) + \
						spin_matrix.item(x, periodic(y,size,-1)) +\
						spin_matrix.item(x, periodic(y,size,1)))
			if np.random.random() <= w[deltaE+8]:
				# Accept!
				spin_matrix[x,y] *= -1 
				M += 2*spin_matrix[x,y] 
				E += deltaE
				
		# Update expectation values 
		E_av 	+= E
		E2_av 	+= E**2
		M_av 	+= M
		M2_av 	+= M**2
		Mabs_av += int(math.fabs(M))
		
	# Normalize average values 
	E_av 		/= float(trials)
	E2_av		/= float(trials) 
	M_av		/= float(trials) 
	M2_av 		/= float(trials)
	Mabs_av 	/= float(trials)
	
	# Calculate variance and normalize to per-point and temp 
	E_variance 	= (E2_av - E_av*E_av)/float(size*size*temp*temp)
	M_variance 	= (M2_av - M_av*M_av)/float(size*size*temp)
	
	# Normalize returned averages to per-point
	E_av		/= float(size*size)
	M_av 		/= float(size*size)
	Mabs_av		/= float(size*size)  
	
	return (E_av, E_variance, M_av, M_variance, Mabs_av)  
	
					
				
