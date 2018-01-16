/**
 * 	@file 	boundary.h
 * 	@brief 	boundary conditions for 2-dim. Ising model as inline function
 * 	@ref	http://en.cppreference.com/w/cpp/language/inline 
 * 	@details Choose correct matrix index with periodic boundary conditions  
 * 
 * 
 * COMPILATION TIP : -Wall warning all, -g debugger, gdb can step through 
 * g++ -Wall -g methodconst.cpp -o methodconst
 * */

#ifndef __BOUNDARY_H__
#define __BOUNDARY_H__  

/**
 * @fn periodic
 * @brief periodic boundary conditions; Choose correct matrix index with 
 * periodic boundary conditions 
 * 
 * Input :
 * @param - i 		: Base index 
 * @param - L 	: Highest \"legal\" index
 * @param - nu		: Number to add or subtract from i
 */
inline int periodic(int i, int L, int nu) {
	return ( i + nu ) % L; // (i + nu) = 0,1,...L-1 
}
	
#endif // END of __BOUNDARY_H__
