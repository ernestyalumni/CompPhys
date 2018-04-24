/**
 * @name 0202_intoverflow.cpp
 * @brief : Program 2.2: A program to illustrate integer overflow  
 * @ref : C++ for Mathematicians: Introduction  
 * 			Program 2.2, pp. 13, Numbers 
 * 
 * */

#include <iostream>  

/**
 * A program to illustrate what happens when large integers
 * are multiplied together.  
 * */
 
int main() {
	int million = 1000000;
	int trillion = million * million ;
	
	std::cout << "According to this computer, " << million << " squared is " 
		<< trillion << "." << std::endl;

}
 
