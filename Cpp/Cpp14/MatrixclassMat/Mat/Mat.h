/** 
 * @file Mat.h
 * @brief C++11 class template for Matrices, ready for Matrix multiplication, 
 * and the taking of the transpose
 * @author Ernest Yeung, <ernestyalumni@gmail.com>
 * @date 20170727
 * */

#ifndef __MAT_H__
#define __MAT_H__

#include <array> // std::array
#include <vector> // std::vector
#include <numeric> // std::inner_product

template<typename Type>
class Mat 
{
	public : 
		/////////////////////////////////////////////
		// Constructors - initializers of the class
		/////////////////////////////////////////////
		Mat(std::array<unsigned int,2> Size_Dims) 
			: Size_Dims_(Size_Dims) 
			{
			// set definitively the size dimensions of our Rows, Columns, and Entries
			Rows_.resize(Size_Dims[0]);
			for (int i=0; i < Size_Dims[0]; i++) { 		
				Rows_[i].resize(Size_Dims[1]); 
				}

			Columns_.resize(Size_Dims[1]);
			for (int j=0; j < Size_Dims[1]; j++) { 		
				Columns_[j].resize(Size_Dims[0]); 
				}
			
			const int number_of_entries = Size_Dims[0]*Size_Dims[1]; 	
			Entries_.resize(number_of_entries); 
			} // END of constructor
			
			
		Mat(unsigned int first_dim, unsigned int second_dim) 
			: Size_Dims_({first_dim,second_dim}) {
			// set definitively the size dimensions of our Rows, Columns, and Entries
			Rows_.resize(first_dim);
			for (int i=0; i < first_dim; i++) { 		
				Rows_[i].resize(second_dim); 
				}

			Columns_.resize(second_dim);
			for (int j=0; j < second_dim; j++) { 		
				Columns_[j].resize(first_dim); 
				}
			
			const int number_of_entries = first_dim*second_dim; 	
			Entries_.resize(number_of_entries); 				
			} // END of constructor

		Mat(std::array<unsigned int,2> Size_Dims, 
				std::vector<Type> & Entries) 
			: Size_Dims_(Size_Dims), Entries_(Entries) {
				const int M = Size_Dims[0]; // M, "number of rows" 
				const int P = Size_Dims[1]; // P, "number of columns"

				// set definitively the size dimensions of our Rows, Columns
				Rows_.resize(M);
				Columns_.resize(P);

				// fill up Rows,Columns from Entries, assumed in "row-major ordering"
				for (int i=0; i<M; i++) {
					for (int j=0; j<P;j++) {
						// idx_global is the index of the entry, if the matrix was 
						// "laid out flat" as a 1-dim. array (i.e. vector)
						const int idx_global = j+P*i; 
						Type entry_input = Entries[idx_global];

						Rows_[i].push_back( entry_input );
						Columns_[j].push_back( entry_input); 
					}
				}

			} // END of constructor

		/** 
		 * @fn Mat = Mat(unsigned int first_dim, unsigned int second_dim, 
		 * 					std::vector<Type> & Entries) 
		 * @brief Constructor (Initializer) for the Mat class template; make Matrices with this
		 * @param Type - class template parameter; keep this in mind to set the entries to be same type		
		 * @param unsigned int first_dim - first size dimension of the matrix
		 * @param unsigned int second_dim - second size dimension of the matrix
		 * @param std::vector<Type> second_dim - entries of the matrix as a 1-dim. std::vector, assuming row-major ordering
		 * 
		 * */
		Mat(unsigned int first_dim, unsigned int second_dim, 
				std::vector<Type> & Entries) 
			: Size_Dims_({first_dim,second_dim}), Entries_(Entries) {				
				const int M = first_dim; // M, "number of rows" 
				const int P = second_dim; // P, "number of columns"

				// set definitively the size dimensions of our Rows, Columns
				Rows_.resize(M);
				Columns_.resize(P);

				// fill up Rows,Columns from Entries, assumed in "row-major ordering"
				for (int i=0; i<M; i++) {
					for (int j=0; j<P;j++) {
						// idx_global is the index of the entry, if the matrix was 
						// "laid out flat" as a 1-dim. array (i.e. vector)
						const int idx_global = j+P*i; 
						Type entry_input = Entries[idx_global];						
						Rows_[i].push_back( entry_input );
						Columns_[j].push_back( entry_input); 
					}
				}
			} // END of constructor

		// following 2 constructors are if we're given a 
		// vector of vectors for the rows
		Mat(std::array<unsigned int,2> Size_Dims, 
				std::vector<std::vector<Type>> & Rows) 
			: Size_Dims_(Size_Dims), Rows_(Rows) {
			const int M = Size_Dims[0]; // M, "number of rows" 
			const int P = Size_Dims[1]; // P, "number of columns"

			// set definitively the size dimensions of our Rows, Columns
			Columns_.resize(P);
			Entries_.resize(M*P);

			// fill up Entries, assuming "row-major ordering" for entries
			for (auto row : Rows_) {
				for (auto entry : row) { 
					Entries_.push_back( entry) ; }
			}
		
			// fill up Columns from Rows, assumed in "row-major ordering"
			for (int i=0; i<M; i++) {						
				for (int j=0; j<P;j++) {
					Type entry_input = (Rows[i])[j];
					Columns_[j].push_back( entry_input); 
				}
			}
		} // END of constructor

		Mat(unsigned int first_dim, unsigned int second_dim, 
				std::vector<std::vector<Type>> & Rows) 
			: Size_Dims_({first_dim,second_dim}), Rows_(Rows) {
			const int M = first_dim; // M, "number of rows" 
			const int P = second_dim; // P, "number of columns"

			// set definitively the size dimensions of our Rows, Columns
			Columns_.resize(P);
			Entries_.resize(M*P);

			// fill up Entries, assuming "row-major ordering" for entries
			for (auto row : Rows_) {
				for (auto entry : row) { 
					Entries_.push_back( entry) ; }
			}
		
			// fill up Columns from Rows, assumed in "row-major ordering"			
			for (int i=0; i<M; i++) {
				for (int j=0; j<P;j++) {		
					Type entry_input = (Rows[i])[j];						
					Columns_[j].push_back( entry_input); 
				}
			}
		} // END of constructor


		// getter functions 
		// get size dimensions of the matrix
		std::array<unsigned int,2> get_size_dims() {
			return Size_Dims_;
		} 
	
		unsigned int get_first_dim() {
			return this->Size_Dims_[0]; 
		}

		unsigned int get_second_dim() {
			return Size_Dims_[1]; 
		}

		// get the (i,j)th entry, i.e. A_{ij}
		Type get_entry(const int i, const int j) {
			return (Rows_[i])[j];
		}

		// get the ith row, i.e. A_{i*}
		std::vector<Type> get_row(const int i) {
			return Rows_[i] ;
		}

		// get the jth column, i.e. A_{*j}
		std::vector<Type> get_column(const int j) {
			return Columns_[j] ;
		}

		// get all the columns of A
		std::vector<std::vector<Type>> get_all_columns() { 
			return Columns_ ;
		}
	
		
		// pretty print functions
		void print_all() {
			for (auto row : Rows_) { 
				for (auto entry : row) {
					std::cout << entry << " ";  }
				std::cout << std::endl; 
			}
		}

		//////////////////////////
		// Matrix Multiplication 
		//////////////////////////
		Mat<Type> operator*(const Mat<Type>& rhs) { 
			// get the size dimensions of the matrices
			// for this M x P matrix (those are its size dimensions)
			unsigned int M = Size_Dims_[0];
			unsigned int P = Size_Dims_[1]; 
			unsigned int rhs_number_of_rows = rhs.Size_Dims_[0]; 
			// N is the number of columns of argument rhs, i.e. 2nd size dim. of the 2nd. matrix to multiply together, with
			unsigned int N = rhs.Size_Dims_[1];
			
			// initialize the rows of resulting matrix C, from this information
			std::vector<std::vector<Type>> C_Rows; 

			// Actual Matrix Multiplication
			auto B_Columns = rhs.Columns_; 
			for (int i = 0; i < M; i++) { 
				std::vector<Type> C_Row; // a row of the resulting matrix C
				auto A_i = Rows_[i];	// A_{i*} 
				for (auto col : B_Columns) { 

					// computing A_{ik} * (B^T)_{jk}
					Type product = std::inner_product( A_i.begin(), 
										A_i.end(), 
										col.begin(), 
										static_cast<Type>(0) );

					C_Row.push_back(product); 
				}
				C_Rows.push_back(C_Row);
			}		

			// initialize the final matrix C
			Mat<Type> C(M,N,C_Rows); 

			return C;				

		} // END of Matrix Multiplication

		//////////////
		// Transpose 
		//////////////
		// T for taking the transpose
		Mat<Type> T() { 
			// reverse the size dimensions
			unsigned int first_dim  = Size_Dims_[1];
			unsigned int second_dim = Size_Dims_[0]; 
			auto new_Rows = Columns_;
			Mat<Type> Atranspose(first_dim,second_dim,new_Rows);
			return Atranspose;  
		}
		

		// destructors - they tell how to end or terminate the class  
		~Mat() {};

		
	private : 
		// size dimensions of the matrix, MxP
		// Size_Dims is (M,P) or number of rows x number of columns; 
		std::array<unsigned int,2> Size_Dims_;

		// vectors of vectors for rows and columns of the Matrix
		// A_{i*}, i =1,2,...M or Size_Dims[0]
		std::vector<std::vector<Type>> Rows_;  
		// A_{*j}, j=1,2,...P or Size_Dims[1]
		std::vector<std::vector<Type>> Columns_; 

		// vector of the entries of the Matrix, in ROW-MAJOR ORDERING 
		std::vector<Type> Entries_;

	
};

#endif // __MAT_H__

	
