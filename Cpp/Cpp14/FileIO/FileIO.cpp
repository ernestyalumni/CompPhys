/**
 * @file   : FileIO.cpp
 * @brief  : File IO, with Python NumPy, in C++14, 
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
 * nvcc -std=c++14 -lcublas ./smartptr/smartptr.cu smartCUBLAS_playground.cu -o smartCUBLAS_playground.exe
 * 
 * */
#include "FileIO.h"


/** @brief Create a class representing a row 
 * 	@ref https://stackoverflow.com/questions/1120140/how-can-i-read-and-parse-csv-files-in-c 
 * */
// member functions
/** @brief operator overload the [] indexing */
		
// operator overloading of indexing
std::string const& CSVRow::operator[](std::size_t index) const 
{	
	return m_data[index];
}

std::size_t CSVRow::size() const
{
	return m_data.size();
}
		
std::vector<std::string> CSVRow::out() {
	return m_data;
}
		
void CSVRow::readNextRow(std::istream& str) 
/** @fn readNextRow
 *  @param std::istream& str - get the address of the 1st byte of memory used to store the string str
 *  */
{
	std::string	line;
			
	/** @details std::getline reads characters from an input stream and places them into a string 
	 *  @ref http://en.cppreference.com/w/cpp/string/basic_string/getline
	 *  @param str - input - the stream to get data from 
	 *  @param line - str - the string to put the data into
	 *  @param delim - the delimiter character
	 * */
	std::getline(str,line);
			
	/** @brief std::stringstream::stringstream - stream class to operate on strings
	 *  @details initialization constructor, (const string& str)
	 *  @param str constructs a stringstream object with a copy of str as content 
	 * 	@ref http://www.cplusplus.com/reference/sstream/stringstream/stringstream/
	 * */
	std::stringstream lineStream(line);
	std::string cell; 
			
	m_data.clear();
	while(std::getline(lineStream, cell, ','))
	{
		m_data.push_back(cell);
	}
	// This checks for a trailing comma with no data after it.
	if (!lineStream && cell.empty())
	{
		// If there was a trailing comma then add an empty element.
		m_data.push_back("");
	}
}		

// operator overload >> for CSVRow_og, specifically
std::istream& operator>>(std::istream& str, CSVRow& data) 
{
	data.readNextRow(str);
	return str;
}

/**
 * 	@fn csv2strvec
 *  @brief .csv filename to vector of vectors of string; 
 */
std::vector<std::vector<std::string>> csv2strvec(std::string & filename) {
	std::ifstream ifs(filename);


	if (!ifs.is_open()) // sanity check to see if the file was even there, otherwise nothing!
	{
		std::cout << "failed to open : " << filename << std::endl; 
	} else {
		CSVRow row;
	
		std::vector<std::vector<std::string>> strvec; // string vector; vector of strings

		while( ifs >> row ) 
		{
			strvec.push_back( row.out() );
		}
		return strvec;
	}
}

/**
 *  @fn pop_header
 *  @brief 	pop 1st element, the header, and remove 1st element from std::vector
 * */
std::vector<std::string> pop_header(std::vector<std::vector<std::string>> &strvec) {
	// pop_front header file
	std::vector<std::string> headerstr = strvec[0];
	
	// remove 1st element
	strvec.erase(strvec.begin());
	
	return headerstr;
}


/**
 *  @fn strvec2fvec
 *  @brief 	vector of vector strings into vector of vector of floats
 * */
std::vector<std::vector<float>> strvec2fvec(std::vector<std::vector<std::string>> &strvec) 
{
	std::vector<std::vector<float>> str2fvec;
	for (auto row : strvec) {
		std::vector<float> frow;
		for (auto ele : row) {
			frow.push_back( std::stof(ele) );
		}
		str2fvec.push_back( frow);
	}
	return str2fvec;
}
	
/* ========== text file (but comma separated); NO HEADER ========== */
/* ========== .csv -> std::vector<std::vector<float>> (directly) ===== */ 
/** @brief Create a class representing a row and store as vector of floats 
 * 	@ref https://stackoverflow.com/questions/1120140/how-can-i-read-and-parse-csv-files-in-c 
 * */
float const& fCSVRow::operator[](std::size_t index) const 
{	
	return frow[index];
}

std::size_t fCSVRow::size() const
{
	return frow.size();
}
		
std::vector<float> fCSVRow::out() {
	return frow;
}
		
void fCSVRow::readNextRow(std::istream& str) 
/** @fn readNextRow
 *  @param std::istream& str - get the address of the 1st byte of memory used to store the string str
 *  */
{
	std::string	line;
			
	/** @details std::getline reads characters from an input stream and places them into a string 
	 *  @ref http://en.cppreference.com/w/cpp/string/basic_string/getline
	 *  @param str - input - the stream to get data from 
	 *  @param line - str - the string to put the data into
	 *  @param delim - the delimiter character
	 * */
	std::getline(str,line);
			
	/** @brief std::stringstream::stringstream - stream class to operate on strings
	 *  @details initialization constructor, (const string& str)
	 *  @param str constructs a stringstream object with a copy of str as content 
	 * 	@ref http://www.cplusplus.com/reference/sstream/stringstream/stringstream/
	 * */
	std::stringstream lineStream(line);
	std::string cell; 
			
	frow.clear();
	while(std::getline(lineStream, cell, ','))
	{
		frow.push_back(std::stof(cell));
	}
	// This checks for a trailing comma with no data after it.
	if (!lineStream && cell.empty())
	{
		// If there was a trailing comma then add an empty element.
//				frow.push_back("");
	}
}


// operator overload >> for fCSVRow, specifically
std::istream& operator>>(std::istream& str, fCSVRow& data) 
{
	data.readNextRow(str);
	return str;
}
	
/**
 * 	@fn csv2fvec
 *  @brief .csv filename to vector of vectors of float
 * 			must assume NO HEADERS
 */
std::vector<std::vector<float>> csv2fvec(std::string & filename) {
	std::ifstream ifs( filename );

	if (!ifs.is_open()) // sanity check to see if file was even there, otherwise nothing!
	{
		std::cout << "failed to open : " << filename << std::endl; 
	} else {
		fCSVRow fcsvrow;
		std::vector<std::vector<float>> fvec; // float vector; vector of floats

		while( ifs >> fcsvrow ) 
		{
			fvec.push_back( fcsvrow.out() );
		}
		return fvec;
	}
} 	

/**
 * 	@fn csv2fvec_hdr
 *  @brief .csv filename to vector of vectors of float
 * 			assume we WANT to throw away the headers
 */
std::vector<std::vector<float>> csv2fvec_hdr(std::string & filename) {
	std::ifstream ifs( filename );

	if (!ifs.is_open()) // sanity check to see if file was even there, otherwise nothing!
	{
		std::cout << "failed to open : " << filename << std::endl; 
	} else {
		fCSVRow fcsvrow;
		std::vector<std::vector<float>> fvec; // float vector; vector of floats

		
		// pop out first row, header
		CSVRow row;
		ifs >> row;

		while( ifs >> fcsvrow ) 
		{
			fvec.push_back( fcsvrow.out() );
		}
		return fvec;
	}
} 	


/* =============== CSVIterator =============== */
/* =============== iterator for a .csv file into strings =============== */

/**
 * 	@ref https://stackoverflow.com/questions/1120140/how-can-i-read-and-parse-csv-files-in-c
 * */
// constructors
CSVIterator::CSVIterator(std::istream& str) : m_str(str.good() ? &str : NULL) 
{
	++(*this); 
}

CSVIterator::CSVIterator() : m_str(NULL) {}
		
// Pre-Increment
CSVIterator& CSVIterator::operator++() {
	if (m_str) { 
		if (!((*m_str) >> m_row)) {
			m_str = NULL; 
		}
	}
	return *this;
}

// Post-Increment
CSVIterator CSVIterator::operator++(int) {
	CSVIterator tmp(*this);
	++(*this);
	return tmp;
}

CSVRow const& CSVIterator::operator*() const {
	return m_row; 
}
		
CSVRow const* CSVIterator::operator->() const {
	return &m_row; 
}
		
bool CSVIterator::operator==(CSVIterator const& rhs) {
	return ((this == &rhs) || (this->m_str == NULL) && (rhs.m_str == NULL)); 
}
		
bool CSVIterator::operator!=(CSVIterator const& rhs) {
	return !((*this) == rhs); 
}

 
/**
 * 	@fn csv2strvecIter
 *  @brief .csv filename to vector of vectors of string
 * 			does the exact same thing as csv2strvecIter, same type of output
 * 			but uses CSVIterator
 */
std::vector<std::vector<std::string>> csv2strvecIter(std::string & filename) 
{
	std::ifstream ifs(filename);

	if (!ifs.is_open()) // sanity check to see if the file was even there, otherwise nothing!
	{
		std::cout << "failed to open : " << filename << std::endl; 
	} else {
		std::vector<std::vector<std::string>> strvec; // string vector; vector of strings

		for (CSVIterator loop(ifs); loop != CSVIterator(); ++loop) 
		{
			auto row = *loop; // CSVRow; use .out() for the vector of strings
			strvec.push_back( row.out() );
		}
	return strvec;
	}	
}	
	

