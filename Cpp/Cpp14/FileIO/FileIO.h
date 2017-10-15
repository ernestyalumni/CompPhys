/**
 * @file   : FileIO.h
 * @brief  : File IO header file, with Python NumPy, in C++14, 
 * @details : class CSVRow 
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
 * nvcc -std=c++14 -lcublas ./smartptr/smartptr.cu smartCUBLAS_playground.cu -o smartCUBLAS_playground.exe
 * 
 * */
#ifndef __FILEIO_H__
#define __FILEIO_H__

#include <string> 	// std::string
#include <fstream> 	// std::ifstream  
#include <vector> 	// std::vector
#include <sstream> 	// std::stringstream

#include <iostream>  

/** @brief Create a class representing a row 
 * 	@ref https://stackoverflow.com/questions/1120140/how-can-i-read-and-parse-csv-files-in-c 
 * */
class CSVRow
{
	private: 
		std::vector<std::string> m_data; // std::vector of strings  

	public:
		/** @brief operator overload the [] indexing */
		std::string const& operator[](std::size_t index) const ;

		std::size_t size() const ;
		
		std::vector<std::string> out() ; 
		
		void readNextRow(std::istream& str) ;
};
		
// operator overload >> for CSVRow, specifically
std::istream& operator>>(std::istream& str, CSVRow& data);	
	
/**
 * 	@fn csv2strvec
 *  @brief .csv filename to vector of vectors of string
 */
std::vector<std::vector<std::string>> csv2strvec(std::string & filename); 


/**
 *  @fn pop_header
 *  @brief 	pop 1st element, the header, and remove 1st element from std::vector
 * */
std::vector<std::string> pop_header(std::vector<std::vector<std::string>> &);

/**
 *  @fn strvec2fvec
 *  @brief vector of vector strings into vector of vector of floats
 * */
std::vector<std::vector<float>> strvec2fvec(std::vector<std::vector<std::string>> &);


/* ========== text file (but comma separated); NO HEADER ========== */
/* ========== .csv -> std::vector<std::vector<float>> (directly) ===== */ 
/** @brief Create a class representing a row and store as vector of floats 
 * 	@ref https://stackoverflow.com/questions/1120140/how-can-i-read-and-parse-csv-files-in-c 
 * */
class fCSVRow
{
	private: 
		std::vector<float> frow; // std::vector of float
	public:
		/** @brief operator overload the [] indexing */
		float const& operator[](std::size_t index) const ;

		std::size_t size() const;
		
		std::vector<float> out() ;
		
		void readNextRow(std::istream& str) ;
		
};

// operator overload >> for fCSVRow, specifically
std::istream& operator>>(std::istream& str, fCSVRow& data); 

/**
 * 	@fn csv2fvec
 *  @brief .csv filename to vector of vectors of float
 * 			must assume NO HEADERS
 */
std::vector<std::vector<float>> csv2fvec(std::string & filename); 


std::vector<std::vector<float>> csv2fvec_hdr(std::string & filename);


/* =============== CSVIterator =============== */
/* =============== iterator for a .csv file into strings =============== */

/**
 * 	@ref https://stackoverflow.com/questions/1120140/how-can-i-read-and-parse-csv-files-in-c
 * */
class CSVIterator
{
	private:
		std::istream*	m_str;
		CSVRow 			m_row;
	public:
		// constructor
		CSVIterator(std::istream& str); 
		
		CSVIterator(); 
		
		// Pre-Increment
		CSVIterator& operator++();

		// Post-Increment
		CSVIterator operator++(int);

	
		CSVRow const& operator*() const; 
		
		CSVRow const* operator->() const; 
		
		bool operator==(CSVIterator const& rhs) ;
		
		bool operator!=(CSVIterator const& rhs) ;
};


/**
 * 	@fn csv2strvecIter
 *  @brief .csv filename to vector of vectors of string using CSVIterator
 * 			does the exact same thing as csv2strvec, same type of output
 * 			but uses CSVIterator
 */
std::vector<std::vector<std::string>> csv2strvecIter(std::string & filename); 





#endif // __FILEIO_H__
