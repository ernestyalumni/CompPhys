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
#include <string> // std::string
#include <fstream> // std::ifstream
#include <vector>   // std::vector
#include <iterator> // std::istream_iterator
#include <iomanip> // std::noskipws

/** @fn readin_single_npyarr
 *  @brief read in a single numpy array
 * */
void readin_single_npyarr(const std::string& filename, 
//    const int no_rows, const int no_columns, 
    std::vector<float> & out_vec) {
    std::ifstream file_in(filename, std::ios::binary);
    if (!file_in.is_open()) {
        std::cout << " failed to open : " << filename << std::endl;
    } else {
        float x;
        while(file_in.read(reinterpret_cast<char*>(&x), sizeof(float))) {
            out_vec.push_back(x);
        }
    }    
};

int main(int argc, char* argv[]) {
    /* 
     * Parameters to MANUALLY change    
     * */
    std::string filename = "./data/A_mat_5_4.npy";

    constexpr const int m = 5; // m = number of rows
    constexpr const int n = 4; // n = number of columns
    /* *** END of Parameters to manually change *** */


    std::ifstream A_5_4_in(filename, std::ios::binary);
    if (!A_5_4_in.is_open()) {
        std::cout << " failed to open : " << filename << std::endl; 
    } else {
        // cf. https://stackoverflow.com/questions/15138353/how-to-read-a-binary-file-into-a-vector-of-unsigned-chars
/*
        // Stop eating new lines in binary mode!!!
        A_5_4_in.unsetf(std::ios::skipws);

        // get its size:
        std::streampos A_5_4_inSize;

        A_5_4_in.seekg(0, std::ios::end);
        A_5_4_inSize = A_5_4_in.tellg();
        A_5_4_in.seekg(0, std::ios::end);
        
        // reserve capacity
//        std::vector<float> A_5_4;
        std::vector<char> A_5_4_charvec;
        std::cout << " A_5_4_inSize : " << A_5_4_inSize << std::endl;
        //      A_5_4.reserve(A_5_4_inSize);
        A_5_4_charvec.reserve(A_5_4_inSize);

        
        // read the data:
        A_5_4_charvec.insert(A_5_4_charvec.begin(),
                        std::istreambuf_iterator<char>(A_5_4_in),
                        std::istreambuf_iterator<char>());

                        
        for (auto ele : A_5_4) {
            std::cout << ele << " ";
        }
        std::cout << std::endl;

        for (auto ele : A_5_4_charvec) {
            std::cout << ele << " ";
        }
        std::cout << std::endl;
        std::cout << A_5_4_charvec[0] << " " << A_5_4_charvec[1] << std::endl; 
*/
        // this WORKS
        // cf. https://stackoverflow.com/questions/19614581/reading-floating-numbers-from-bin-file-continuosly-and-outputting-in-console-win
        std::vector<float> A_5_4;
        float x;
        while (A_5_4_in.read(reinterpret_cast<char*>(&x), sizeof(float))) {
//            std::cout << x << std::endl;
            A_5_4.push_back(x);
        }
        for (auto ele : A_5_4) {
            std::cout << ele << " ";
        }
        std::cout << std::endl;


    } // END of if-else check if filename did exist
    std::vector<float> A_5_4_b;
    
    readin_single_npyarr(filename, A_5_4_b);
    for (auto ele : A_5_4_b) {
        std::cout << ele << " ";
    }
    std::cout << std::endl;


    return EXIT_SUCCESS;
}
