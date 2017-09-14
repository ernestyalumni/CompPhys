/**
 * @file   : FileIObin.cpp
 * @brief  : Examples to demonstrate binary file IO in C/C++, especially C++11/14
 * @author : Ernest Yeung <ernestyalumni@gmail.com>
 * @date   : 20170913
 * @ref    : https://stackoverflow.com/questions/11563963/writing-a-binary-file-in-c-very-fast
 * 				Writing a binary file in C++ very fast
 * 
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
 * g++ -std=c++14 FileIObin.cpp -o FileIObin.exe
 * 
 * */
 /*
#include <stdio.h>
const unsigned long long size = 8ULL*1024ULL*1024ULL;
unsigned long long a[size];

int main() 
{
	FILE* pFILE;
	pFile = fopen("file.bin", "wb");
	for (unsigned long long j=0;j<1024; ++j) {
		// 
*/

#include <vector> 		// std::vector
#include <cstdint>		// uint64_t, std::size_t 

#include <fstream>		// std::fstream
#include <chrono>
#include <numeric>		// std::iota
#include <random>		// std::shuffle, std::mt19937, std::random_device
#include <algorithm>
#include <iostream>
#include <cassert>		// assert


std::vector<uint64_t> GenerateData(std::size_t bytes)
{
	assert(bytes % sizeof(uint64_t) == 0);
	std::vector<uint64_t> data(bytes / sizeof(uint64_t));
	std::iota(data.begin(), data.end(),0);	// fills the range [first, last) with sequentially increasing values, starting with value 0
	std::shuffle(data.begin(),data.end(), std::mt19937{ std::random_device{}() });
	return data;		
}

long long option_1(std::size_t bytes) 
{
	std::vector<uint64_t> data = GenerateData(bytes);
	
	auto startTime = std::chrono::high_resolution_clock::now();
	auto myfile = std::fstream("file.binary", std::ios::out | std::ios::binary);
	myfile.write((char*)&data[0],bytes);
	myfile.close();
	auto endTime = std::chrono::high_resolution_clock::now();
	
	return std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
}

long long option_2(std::size_t bytes)
{
	std::vector<uint64_t> data = GenerateData(bytes);
	
	auto startTime = std::chrono::high_resolution_clock::now();
	FILE* file = fopen("file.binary", "wb");
	fwrite(&data[0], 1, bytes, file);
	fclose(file);
	auto endTime = std::chrono::high_resolution_clock::now();
	
	return std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
}
	
long long option_3(std::size_t bytes)
{
	std::vector<uint64_t> data = GenerateData(bytes);
	
	std::ios_base::sync_with_stdio(false);
	auto startTime = std::chrono::high_resolution_clock::now();
	auto myfile = std::fstream("file.binary", std::ios::out | std::ios::binary);
	myfile.write((char*)&data[0], bytes);
	myfile.close();
	auto endTime = std::chrono::high_resolution_clock::now();
	
	return std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
}

int main()
{
	const std::size_t kB = 1024;
	const std::size_t MB = 1024 * kB;
	const std::size_t GB = 1024 * MB;  

	for (std::size_t size = 1 * MB; size <= 4 * GB; size *= 2) std::cout << "option1, " << size / MB << "MB: " << option_1(size) << "ms" << std::endl;  
	for (std::size_t size = 1 * MB; size <= 4 * GB; size *= 2) std::cout << "option2, " << size / MB << "MB: " << option_2(size) << "ms" << std::endl;  
	for (std::size_t size = 1 * MB; size <= 4 * GB; size *= 2) std::cout << "option3, " << size / MB << "MB: " << option_3(size) << "ms" << std::endl;  

	return 0;
}
