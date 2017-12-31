/**
 * @file   : ptr2bin.cpp
 * @brief  : smart ptr to binary file (i.e. output, File I/O) 
 * @details : smart ptr to binary file 
 * @author : Ernest Yeung <ernestyalumni@gmail.com>
 * @date   : 20171231    
 * @ref    : http://en.cppreference.com/w/cpp/io/basic_fstream
 * Ch. 38 Bjarne Stroustrup, The C++ Programming Language, 4th Ed. 2013 Addison-Wesley 
 * 
 * https://www.paypal.com/cgi-bin/webscr?cmd=_donations&business=ernestsaveschristmas%2bpaypal%40gmail%2ecom&lc=US&item_name=ernestyalumni&currency_code=USD&bn=PP%2dDonationsBF%3abtn_donateCC_LG%2egif%3aNonHosted 
 * 
 * which won't go through a 3rd. party such as indiegogo, kickstarter, patreon.  
 * Otherwise, I receive emails and messages on how all my (free) material on 
 * physics, math, and engineering have helped students with their studies, 
 * and I know what it's like to not have money as a student, but love physics 
 * (or math, sciences, etc.), so I am committed to keeping all my material 
 * open-source and free, whether or not 
 * sufficiently crowdfunded, under the open-source MIT license: 
 * 	feel free to copy, edit, paste, make your own versions, share, use as you wish.  
 *  Just don't be an asshole and not give credit where credit is due.  
 * Peace out, never give up! -EY
 * 
 * */
/* 
 * COMPILATION TIP
 * g++ main.cpp ./structs/structs.cpp -o main
 * 
 * */

#include <iostream> 
#include <fstream> // std::fstream  
#include <string>  // std::string  

#include <memory> // std::unique_ptr, std::make_unique  


void ptr2bin( std::string& filename, std::unique_ptr<double[]>& uptr, size_t N) {
//	std::fstream fstre(filename, std::ios::binary); 
	std::fstream fstre(filename, fstre.binary | fstre.trunc | fstre.in | fstre.out ); 
	if (!fstre.is_open()) {
		std::cout << "Failed to open " << filename << std::endl; 
	} else {
		for (size_t idx=0; idx<N; idx++) {
			double input = uptr[idx];  
			fstre.write(reinterpret_cast<char*>(&input), sizeof input); // binary output 
		}
		fstre.close();  
	}  
};


int main() {
	std::string filename = "test1.bin";  

	/* boilerplate, test values */
	size_t N = 10;
	std::unique_ptr<double[]> test_uptr = std::make_unique<double[]>(N); 
	for (int idx=0; idx<N; idx++) {
		test_uptr[idx] = ((double) idx + 1); 
		std::cout << test_uptr[idx] << " ";
	} std::cout << std::endl << std::endl; 
	
	ptr2bin(filename, test_uptr,N); 

	// sanity check; read file 
	// read
	std::unique_ptr<double[]> read_uptr = std::make_unique<double[]>(N); 
	std::ifstream fread(filename, std::ios::binary ); 
	
	for (size_t idx=0; idx<N; idx++) {
		double output;
		fread.read(reinterpret_cast<char*>(&output), sizeof output);  
		std::cout << " " << output; 
	}

/*
	std::fstream fstre(filename, std::ios::binary);  
	if (!fstre.is_open())  {
		std::cout << "Failed to open " << filename << std::endl; 
	} else {
		
		
	}
*/	
	
	
	
}
