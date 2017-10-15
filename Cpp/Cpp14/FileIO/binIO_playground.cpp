/**
 * @file   : binIO_playground.cpp
 * @brief  : Binary IO playground with Python NumPy and CUBLAS, in C++14, 
 * @details : A playground to try out things with binary I/O, files saved in binary format; 
 * 				especially abstracting our use of smart pointers with CUDA.  
 * 				use FLAG std::ios::binary 
 * 			cf. https://stackoverflow.com/questions/6488847/read-entire-binary-file-into-an-array-in-single-call-c
 * 				https://stackoverflow.com/questions/37503346/writing-binary-in-c-and-read-in-python
 * 
 * @author : Ernest Yeung <ernestyalumni@gmail.com>
 * @date   : 20171013  
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
#include <string> 	// std::string
#include <fstream> // std::ifstream
#include <vector> 	// std::vector
#include <sstream> // std::stringstream
#include <memory>	// std::unique_ptr
#include <iterator> 	// std::istreambuf_iterator

/** @brief Create a class representing a row 
 * 	@ref https://stackoverflow.com/questions/1120140/how-can-i-read-and-parse-csv-files-in-c 
 * */
class CSVRow_og
{
	private: 
		std::vector<std::string> m_data; // std::vector of strings  
	public:
		/** @brief operator overload the [] indexing */
		std::string const& operator[](std::size_t index) const 
		{	
			return m_data[index];
		}
		std::size_t size() const
		{
			return m_data.size();
		}
		
		std::vector<std::string> out() {
			return m_data;
		}
		
		void readNextRow(std::istream& str) 
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
		
};

/** @brief Create a class representing a row and store as vector of floats 
 * 	@ref https://stackoverflow.com/questions/1120140/how-can-i-read-and-parse-csv-files-in-c 
 * */
class fCSVRow
{
	private: 
		std::vector<float> frow; // std::vector of float
	public:
		/** @brief operator overload the [] indexing */
		float const& operator[](std::size_t index) const 
		{	
			return frow[index];
		}
		std::size_t size() const
		{
			return frow.size();
		}
		
		std::vector<float> out() {
			return frow;
		}
		
		void readNextRow(std::istream& str) 
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
		
};


// operator overload >> for CSVRow_og, specifically
std::istream& operator>>(std::istream& str, CSVRow_og& data) 
{
	data.readNextRow(str);
	return str;
}

// operator overload >> for fCSVRow, specifically
std::istream& operator>>(std::istream& str, fCSVRow& data) 
{
	data.readNextRow(str);
	return str;
}

/**
 * 	@ref https://stackoverflow.com/questions/1120140/how-can-i-read-and-parse-csv-files-in-c
 * */
class CSVIterator_og
{
	private:
		std::istream*	m_str;
		CSVRow_og 		m_row;
	public:
		// constructor
		CSVIterator_og(std::istream& str) : m_str(str.good() ? &str : NULL) {
			++(*this); 
		}
		CSVIterator_og() : m_str(NULL) {}
		
		// Pre-Increment
		CSVIterator_og& operator++() {
			if (m_str) { 
				if (!((*m_str) >> m_row)) {
					m_str = NULL; 
				}
			}
			return *this;
		}

		// Post-Increment
		CSVIterator_og operator++(int) {
			CSVIterator_og tmp(*this);
			++(*this);
			return tmp;
		}

		CSVRow_og const& operator*() const {
			return m_row; 
		}
		
		CSVRow_og const* operator->() const {
			return &m_row; 
		}
		
		bool operator==(CSVIterator_og const& rhs) {
			return ((this == &rhs) || (this->m_str == NULL) && (rhs.m_str == NULL)); 
		}
		
		bool operator!=(CSVIterator_og const& rhs) {
			return !((*this) == rhs); 
		}
};

/**
 * 	@brief iterator for a .csv file of floats (that are strings)
 * */
class fCSVIterator 
{
	private:
		std::unique_ptr<std::istream> 	u_str; // std::unique_ptr to a std::istream
		fCSVRow 	frow; 	// want: row of floats
	public:
		// constructors
//		fCSVIterator(std::istream& str) : 
//		u_str = std::move(str_u);  

		fCSVIterator() : u_str(nullptr) {} 
		
		// Pre-Increment
/*		fCSVIterator& operator++() {
			if (u_str) {
				if (!((u_str.get() >> frow)) {
					u_str
*/
};

int main(int argc, char* argv[]) {
    /* 
     * Parameters to MANUALLY change    
     * */
	std::string filename = "./data/A_mat_5_4.npy";
	
//    constexpr const int m = 5; // m = number of rows
//    constexpr const int n = 4; // n = number of columns
    int m = 5; // m = number of rows
    int n = 4; // n = number of columns

	/* *** END of Parameters to manually change *** */	

	std::ifstream ifs_filein(filename, std::ios::binary);
	if (!ifs_filein.is_open()) {
		std::cout << "failed to open : " << filename << std::endl; 
	} else {
		std::vector<float> A;

		float x;
		while (ifs_filein.read(reinterpret_cast<char*>(&x), sizeof(float))) {
			/* &x gets the address of the first byte of memory used to store the object
			 * reinterpret_cast<char*> treats that memory as bytes, threat a float as a sequence of bytes
			 * */
			std::cout << x << std::endl;
			A.push_back(x);
		}
	
	
		/* ===== Row-major ordering ===== */ 
		// Assuming row-major ordering for the flattened matrix A 
	
		for (int i = 0; i<m; i++) {
			for (int j =0; j<n; j++) {
				std::cout << A[i*n+j] << " ";  }
			std::cout << std::endl; 
		}
	}	


	/* ===== manuf_learn.dat - whitespace separated files ===== */ 
	// Do data preprocessing with Pandas
	// output to numpy array binary  

	std::string filename_manuf_learn = "./data/manuf_learn.npy"; 
	m=20;
	n=8;
	
	std::ifstream ifs_manuf(filename_manuf_learn, std::ios::binary);
	if (!ifs_manuf.is_open()) {
		std::cout << "failed to open : " << filename_manuf_learn << std::endl; 
	} else {
		std::vector<float> A;

		float x;
		while (ifs_manuf.read(reinterpret_cast<char*>(&x), sizeof(float))) {
			/* &x gets the address of the first byte of memory used to store the object
			 * reinterpret_cast<char*> treats that memory as bytes, threat a float as a sequence of bytes
			 * */
			std::cout << x << std::endl;
			A.push_back(x);
		}
	
	
		/* ===== Row-major ordering ===== */ 
		// Assuming row-major ordering for the flattened matrix A 
	
		for (int i = 0; i<m; i++) {
			for (int j =0; j<n; j++) {
				std::cout << A[i*n+j] << " ";  }
			std::cout << std::endl; 
		}
	}	

	/* =============== .csv ================= */ 
	/* ========== .csv -> std::ifstream ===== */ 
	// Do data preprocessing with Pandas
	// output to numpy array binary  
	std::string filename_copoly = "./data/copolymer_viscosity.csv";
	std::ifstream ifs_copoly( filename_copoly );
	
	std::cout << " ifs_copoly : " << ifs_copoly.is_open() << std::endl;
	
	CSVRow_og row_og;
	
	std::vector<std::vector<std::string>> copoly_strvec; // string vector; vector of strings

	while( ifs_copoly >> row_og ) 
	{
		copoly_strvec.push_back( row_og.out() );
		std::cout << " 2nd element (" << row_og[1] << ") " << std::endl;
	}
	
	// pop_front header file 
	std::vector<std::string> copoly_headerstr= copoly_strvec[0];
	for (auto hdr : copoly_headerstr) { std::cout << hdr << " "; }
	std::cout << std::endl << " END of copolymer header " << std::endl;
	
	// remove 1st element
	copoly_strvec.erase(copoly_strvec.begin());
	
	std::vector<std::vector<float>> copoly_str2fvec;
	for (auto row : copoly_strvec) {
		std::vector<float> frow;
		for (auto ele : row) {
			frow.push_back( std::stof(ele) );
		}
		copoly_str2fvec.push_back( frow);
	}
	std::cout << copoly_strvec[0][1] << std::endl;

	std::cout << std::endl << " Copolymer viscosity as vector of floats : " << std::endl;
	for (auto ele : copoly_str2fvec[0]) {
		std::cout << ele << " "; }	std::cout << std::endl;

	std::vector<float> test_vec(5,1.0f);
//	std::cout << test_vec[2] << std::endl;
//	test_vec.push_back(NULL); // warning: passing NULL to non-pointer argument 1

	/* ========== text file (but comma separated); NO HEADER ========== */
	std::string filename_ex2data1 = "./data/ex2data1.txt";
	std::ifstream ifs_ex2data1( filename_ex2data1 );
	fCSVRow fcsvrow;
	std::vector<std::vector<float>> ex2data1_fvec; // float vector; vector of floats
	bool header_flag = false;

	std::cout << " ifs_ex2data1.is_open() : " << ifs_ex2data1.is_open() << std::endl;
	while( ifs_ex2data1 >> fcsvrow ) 
	{
//		if (header_flag) {
	//		header_flag = false;
		//} else {
		ex2data1_fvec.push_back( fcsvrow.out() );
		
	}

	std::cout << " ex2data1, first 3 columns " << std::endl;
	for (auto row : ex2data1_fvec) {
		std::cout << row[0] << " " << row[1] << " " << row[2] << std::endl; 
	}
	std::cout << ex2data1_fvec.size() << std::endl;

	/* ===== using fCSVIterator with no header, comma-separated txt file ===== */
	fCSVIterator test_iter;
	
	std::ifstream ifs2_ex2data1( filename_ex2data1 );
	
	std::istreambuf_iterator<char> i1(ifs2_ex2data1); 
	
	while (i1 !=std::istreambuf_iterator<char>()) {
		char c = *i1++;
		std::cout << c; 
	}
	
	std::cout << std::endl << " ===== END of ifs2_ex2data1 ===== " << std::endl;
	
	/* ===== text file (but comma separated); HEADER ===== */
	std::string filename_rockstr = "./data/rockstrength.csv";
	std::ifstream ifs_rockstr( filename_rockstr );
	std::vector<std::vector<float>> rockstr_fvec; // float vector; vector of floats
	std::cout << " ifs_rockstr.is_open() : " << ifs_rockstr.is_open() << std::endl;
	ifs_rockstr >> row_og; 
	std::vector<std::string> rockstr_header = row_og.out(); 
	std::cout << row_og[0] << std::endl;
//	auto rockstr_header = row_og.out();
	for (auto ele : rockstr_header) { std::cout << ele << " "; } std::cout << std::endl;
	while( ifs_rockstr >> fcsvrow ) 
	{
		rockstr_fvec.push_back( fcsvrow.out() );		
	}
	for (int row=0; row < 3; row++) {
		std::cout << rockstr_fvec[row][0] << " " << rockstr_fvec[row][1] 
			<< " " << 	rockstr_fvec[row][2] << " " << rockstr_fvec[row][3]; 
		std::cout << std::endl;
	}

	/* ===== using CSVIterator_og ===== */
	std::ifstream ifs_rockstr2( filename_rockstr );

	for (CSVIterator_og loop(ifs_rockstr2); loop != CSVIterator_og(); ++loop) 
	{
		std::cout << (*loop)[0] << " " << (*loop)[1] << " " << (*loop)[2] << " " <<
			(*loop)[3] << " " << (*loop)[4] << std::endl;
	}

	
	
	return EXIT_SUCCESS;
}
