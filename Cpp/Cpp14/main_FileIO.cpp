/**
 * @file   : main_FileIO.cpp
 * @brief  : "main" file for File IO, with Python NumPy, in C++14, 
 * @details :  class CSVRow
 * 
 * 
 * @author : Ernest Yeung <ernestyalumni@gmail.com>
 * @date   : 20171014  
 * @ref    : 
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
 * nvcc -std=c++14 -lcublas ./FileIO/FileIO.cpp main_FileIO.cpp -o main_FileIO.exe
 * 
 * */
#include "./FileIO/FileIO.h"
#include <iostream> 	// std::cout

int main(int argc, char* argv[]) {


	/* =============== .csv ================= */ 
	/* ========== .csv -> std::ifstream ===== */ 
	// ===== std::ifstream -> CSVRow ('s, iteratively many CSVRows) ===== */
	// Use CSVRow do read in each row from std::ifstream

	std::string filename_copoly = "./FileIO/data/copolymer_viscosity.csv";
	std::ifstream ifs_copoly( filename_copoly );
	
	std::cout << " ifs_copoly : " << ifs_copoly.is_open() << std::endl;
	
	CSVRow row;
	
	std::vector<std::vector<std::string>> copoly_strvec; // string vector; vector of strings

	while( ifs_copoly >> row ) 
	{
		copoly_strvec.push_back( row.out() );
		std::cout << " 2nd element (" << row[1] << ") " << std::endl;
	}

	// sanity check, print out copoly_strvec
	for (auto ele : copoly_strvec[0]) {
		std::cout << ele << " "; }	std::cout << std::endl;

	for (int j=0;j<3;j++) {
		std::cout << copoly_strvec[1][j] << " "; 
	}


	// use defined function to wrap up previous steps
	auto copoly_strvec2 = csv2strvec(filename_copoly);
	
	// sanity check, print out copoly_strvec2
	for (auto ele : copoly_strvec2[0]) {
		std::cout << ele << " "; }	std::cout << std::endl;

	for (int j=0;j< 4 ; j++) {
		std::cout << copoly_strvec2[1][j] << " "; 
	}
	std::cout << std::endl << copoly_strvec2[1].size() << std::endl;

	std::cout << std::endl;

	// ===== std::vector<std::string> header removal ===== 
	// use defined function to wrap up steps
	auto copoly_hdr = pop_header(copoly_strvec2);
	for (auto hdr : copoly_hdr) { std::cout << hdr << " "; }
	std::cout << std::endl << " END of copolymer header " << std::endl;

	// ===== strvec2fvec ===== //
	// string vector to vector of floats
	auto copoly_str2fvec = strvec2fvec(copoly_strvec2);
	
	std::cout << std::endl << " Copolymer viscosity as vector of floats : " << std::endl;
	for (auto ele : copoly_str2fvec[0]) {
		std::cout << ele << " "; }	std::cout << std::endl;
	for (auto ele : copoly_str2fvec[1]) {
		std::cout << ele << " "; }	std::cout << std::endl;


	/* ========== text file (but comma separated); NO HEADER ========== */
	std::string filename_ex2data1 = "./FileIO/data/ex2data1.txt";

	auto ex2data1_fvec = csv2fvec(filename_ex2data1);

	std::cout << " ex2data1, first 3 columns " << std::endl;
	for (auto row : ex2data1_fvec) {
		std::cout << row[0] << " " << row[1] << " " << row[2] << std::endl; 
	}


	/* ========== text file (but comma separated); HEADER ========== */
	
	/* ===== using CSVIterator ===== */
	std::string filename_rockstr = "./FileIO/data/rockstrength.csv";

	auto rockstr_strvec = csv2strvecIter( filename_rockstr );
	auto rockstr_hdr = pop_header(rockstr_strvec);
	auto rockstr_str2fvec = strvec2fvec(rockstr_strvec);
	
	// sanity check, print out rockstr_str2fvec
	for (auto hdr : rockstr_hdr) { std::cout << hdr << " " ; } std::cout << std::endl;
	for (auto ele : rockstr_str2fvec[0]) {
		std::cout << ele << " "; }	std::cout << std::endl;

	for (int j=0;j< 4 ; j++) {
		std::cout << rockstr_str2fvec[1][j] << " "; 
	}  std::cout << std::endl; 
		
	// also to show if one just wants to throw away the header	
	auto rockstr_fvec_NoHeaders = csv2fvec_hdr(filename_rockstr);

	for (auto ele : rockstr_fvec_NoHeaders[0]) {
		std::cout << ele << " "; }	std::cout << std::endl;

	for (int j=0;j< 4 ; j++) {
		std::cout << rockstr_fvec_NoHeaders[1][j] << " "; 
	}  std::cout << std::endl; 



}


