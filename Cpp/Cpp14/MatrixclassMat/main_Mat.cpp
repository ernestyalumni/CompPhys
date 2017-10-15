/** 
 * @file main_Mat.cpp
 * @brief "Main" file to demonstrate the use of class template Mat 
 * @author Ernest Yeung, <ernestyalumni@gmail.com>
 * @date 20170727
 * @details Compile this source file like this:  
 * 	g++ -std=c++11 main_Mat.cpp 
 * 			Similarly, when you use the header file Mat.h in your own code, 
 * 			simply add the include (i.e. #include) line for the header file, 
 * 			with the correct path.  
 *			One can also copy the header file into the include/ directory in 
 * 			"root" directory (usually this is /usr/) as an administrator and 
 * 			include the class template Mat in Mat.h by the flag -lMat: 
 * 	g++ -std=c++11 main_Mat.cpp -lMat
 * 
 * */
#include <iostream> // std::cout
#include <array> // std::array
#include <vector> // std::vector

// These includes are to make "boilerplate" examples
#include <algorithm> // std::for_each

#include "./Mat/Mat.h"  // Mat

int main() 
{
	std::cout << "\n Let us demonstrate the usage of the class template Mat, \n " <<
					" \t for matrix multiplication, and the transpose of a matrix " <<
						std::endl;  
						
	// Examples from Matrix Multiplication, the human way! Article in AMS blog
	// cf. http://blogs.ams.org/mathgradblog/2017/06/12/matrix-multiplication-human-way/
	std::cout << "\n You can initialize a matrix with its size dimensions by either \n " <<
					" \t inputing a 2-dim. std::array with the number of rows and \n " <<
					" \t number of columns (in that order) " << std::endl;  
					
	std::array<unsigned int, 2> SizeDims_Example0 {3,3}; 
	std::cout << "\n Size Dimensions as an std::array : " << 
		SizeDims_Example0[0] << " x " << SizeDims_Example0[1] << std::endl; 
	
	std::cout << "\n Or as 2 separate integers or unsigned integers : " << std::endl; 
	const int number_of_rows_Example1 = 3; 
	const int number_of_columns_Example1 = 3;  
	std::cout << "\n Number of rows for Example 1 : " << number_of_rows_Example1 
				<< std::endl; 
	std::cout << "\n Number of columns for Example 1 : " << number_of_columns_Example1 
				<< std::endl; 

	std::cout << "\n You can choose the type of the matrix entries \n " << 
					" (mathematically, this is choosing the 'field' that \n " << 
					" the matrix entries belong to, whether they're real numbers \n " <<
					" or integers, for example) by specifying the class template's \n " << 
					" template parameter for the class template Mat right from the \n " <<
					" beginning when using it. " << std::endl; 
										
	std::cout << "\n You input the entries into a (flattened) 1-dim. std::vector \n " << 
					" containing the entries in the so-called ROW-MAJOR ORDER, \n " << 
					" i.e. order the elements in each row in sequence, and then \n " <<
					" the next row, and the next, and so on. " << std::endl; 
				
	std::vector<int> Example0_entries { 1,2,3,4,5,6,7,-8,0};

	// initialize the matrix with its entries 
	Mat<int> Example0(SizeDims_Example0, Example0_entries); 
	
	std::cout << "\n Use the .print_all() class method to pretty print the matrix " << std::endl;
	std::cout << " \n Here is Example 0 3x3 matrix: " << std::endl; 
	Example0.print_all(); 
						
	std::vector<int> Example1_entries { 0,0,0,0,1,0,1,0,0};

	// initialize the matrix with its entries 
	Mat<int> Example1(number_of_rows_Example1, number_of_columns_Example1, 
						Example1_entries); 
	std::cout << " \n Here is Example 1 3x3 matrix : " << std::endl; 		
	Example1.print_all();
				
	std::cout << "\n Multiply Example 0 and Example 1 matrices together: \n " << std::endl; 
	auto Example01 = Example0 * Example1;
	Example01.print_all(); 

	std::cout << "\n We can also take the TRANSPOSE of this matrix, " << 
					" \n and the other matrices before with class template Mat; " << 
					" \n just use the .T() class method. " << std::endl;
	(Example01.T()).print_all();

	
	std::cout << "\n You can also input in the entries of the matrix as a \n " << 
				 " std::vector of std::vector 's, but this may be more \n " << 
				 " cumbersome than simply inputting in a 1-dim. std::vector, \n " << 
				" assuming so-called ROW-MAJOR ordering. \n " << std::endl;  

	std::vector<float> Example2_row0 { 1., -5. , 3. } ; 
	std::vector<float> Example2_row1 { 0., -0.5 , 2. } ; 
	std::vector<float> Example2_row2 { 1., -8. , 0. } ; 
					
	std::vector<std::vector<float>> Example2_entries { Example2_row0, 
														Example2_row1, 
														Example2_row2 };

	// initialize the matrix with its entries already in rows
	Mat<float> Example2(3,3,Example2_entries);  				
						
	std::cout << "\n Example 2 matrix : " << std::endl; 													
	Example2.print_all(); 

	std::cout << "\n Let's try taking the transpose of this matrix : " << std::endl;
	(Example2.T()).print_all();

	std::vector<float> Example3_entries { 2,-1,0 };
	Mat<float> Example3(3,1,Example3_entries); 
	std::cout << "\n Example 3 vector, as a matrix Mat : " << std::endl ; 
	Example3.print_all(); 

	std::cout << "\n Let's try taking the transpose of this 'column vector' : " << std::endl;
	(Example3.T()).print_all();  

	std::cout << "\n Matrix multiply them together (i.e. take the matrix product of the 2) : " << std::endl ;  
	(Example2 * Example3).print_all();
	
	
	// Miscellaneous examples
	std::cout << "\n I will show some more examples, with various " <<
					"\n matrix size dimensions, of matrix multiplication " <<
					" \n and the transpose \n" << std::endl;
	
	std::vector<int> A4_entries { 1,2,3,4,5,-2,7,-4,0}; 
	std::vector<int> B4_entries {0,1,3};
	Mat<int> A4(3,3,A4_entries); 
	Mat<int> B4(3,1,B4_entries); 
	A4.print_all(); 
	std::cout << " \n * \n " << std::endl; 
	B4.print_all(); 
	std::cout << " \n ====> \n " << std::endl; 
	(A4 * B4).print_all(); 

	std::cout << "\n Its transpose : " << std::endl; 
	((A4 * B4).T()).print_all(); 

	std::cout << "\n transpose of the vector : " << std::endl; 
	(B4.T()).print_all();  
	
	std::cout << " \n Even more examples (follow the code itself to see what were " << 
					" \n the operations taken, exactly, but I demonstrate " <<
					" \n matrix multiplication and transpose of various matrices " <<
					" \n below. " << std::endl;

	std::vector<float> A5_entries { 1,2,3,4,5,6,7,8,9,10,11,12};
	Mat<float> A5(4,3,A5_entries); 
	std::cout << "\n Matrix 'A' : \n " << std::endl; 
	A5.print_all(); 
	std::cout << "\n Its transpose: \n " << std::endl; 
	(A5.T()).print_all();  

	std::vector<float> B5_entries { 11,22,33,44,55,66} ; 
	Mat<float> B5(3,2,B5_entries); 
	std::cout << "\n Matrix 'B' : \n " << std::endl; 
	B5.print_all();
	std::cout << "\n Its transpose: \n " << std::endl; 
	(B5.T()).print_all(); 

	std::cout << "\n Multiply them together \n " << std::endl; 
	(A5*B5).print_all();

	std::cout << "\n Take this product's transpose \n" << std::endl; 
	((A5*B5).T()).print_all(); 

	std::cout << "\n Note that I assume the user knows some linear algebra " << 
					"\n and will make sure to have the 'side' matrix size dimensions " << 
					"\n match for matrix multiplication to make sense.  " << 
					"\n Let's see what happens when the 'inner' size dims. " <<
					"\n aren't equal.  " << std::endl; 
	
	(A5*(B5.T())).print_all(); 

	// Try out "big" or "large" matrices
	std::cout << " \n Let's try out 'large' matrices. Let's make up a few examples \n" << std::endl; 

	const int N = 10;
	// "boilerplate" to make an interesting matrix of 3 x N size dimensions.  
	std::vector<float> range_row;
	for (int idx =1; idx <=N; idx++) { range_row.push_back(static_cast<float>(idx)); }
	std::cout << range_row.size() << std::endl; 

	auto range_row2 = range_row;
	std::for_each( range_row2.begin(), range_row2.end(), [](float &ele) { ele=ele*ele; });
	auto range_row3 = range_row;
	std::for_each( range_row3.begin(), range_row3.end(), [](float &ele) { ele=ele*ele*ele; });

	std::vector<std::vector<float>> Abig_entries { range_row,range_row2,range_row3 };

	Mat<float> Abig(3,N,Abig_entries);
	Abig.print_all();

	std::vector<std::vector<float>> Bbig_entries { range_row,range_row,range_row };
	Mat<float> Bbig(3,N,Bbig_entries);

	// This should reproduce the summation identities found on wikipedia.org: 
	// https://en.wikipedia.org/wiki/Summation#Identities
	(Abig * (Bbig.T())).print_all();

	// Let's try an even bigger matrix:
	
	const int N1 = 1000;
	// "boilerplate" to make an interesting matrix of 3 x N size dimensions.  
	std::vector<float> range_row1;
	for (int idx =1; idx <=N1; idx++) { range_row1.push_back(static_cast<float>(idx)); }
	std::cout << range_row1.size() << std::endl; 

	auto range_row12 = range_row1;
	std::for_each( range_row12.begin(), range_row12.end(), [](float &ele) { ele=ele*ele; });
	auto range_row13 = range_row1;
	std::for_each( range_row13.begin(), range_row13.end(), [](float &ele) { ele=ele*ele*ele; });

	std::vector<std::vector<float>> Alarge_entries { range_row1,range_row12,range_row13 };

	Mat<float> Alarge(3,N1,Alarge_entries);

	std::vector<std::vector<float>> Blarge_entries { range_row1,range_row1,range_row1 };
	Mat<float> Blarge(3,N1,Blarge_entries);

	// This should reproduce the summation identities found on wikipedia.org: 
	// https://en.wikipedia.org/wiki/Summation#Identities
	(Alarge * (Blarge.T())).print_all();
	


}


