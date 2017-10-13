/**
 * @file   : binIO_playground.cu
 * @brief  : Binary IO playground with CUBLAS, in C++14, 
 * @details : A playground to try out things with binary I/O, files saved in binary format; 
 * 				especially abstracting our use of smart pointers with CUDA.  
 * 				use FLAG std::ios::binary 
 * 				Notice that std::make_unique DOES NOT have a custom deleter! (!!!)
 * 				Same with std::make_shared!  
 * 			cf. https://stackoverflow.com/questions/6488847/read-entire-binary-file-into-an-array-in-single-call-c
 * 				https://stackoverflow.com/questions/37503346/writing-binary-in-c-and-read-in-python
 * 
 * @author : Ernest Yeung <ernestyalumni@gmail.com>
 * @date   : 20171009  
 * @ref    : cf. Peter Gottschling. 
 * 		Discovering Modern C++: An Intensive Course for Scientists, Engineers, and Programmers, A.2.7 Binary I/O. 
 * If you find this code useful, feel free to donate directly and easily at this direct PayPal link: 
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
 * nvcc -std=c++14 -lcublas ./smartptr/smartptr.cu smartCUBLAS_playground.cu -o smartCUBLAS_playground.exe
 * 
 * */
#include <iostream> // std::cout

#include <fstream> // std::ifstream

int main(int argc, char* argv[]) {
	std::ifstream A_5_4_in;
	A_5_4_in.open("./data/A_mat_5_4.npy",std::ios::binary); // std::ios::binary (for binary files)

	
}
