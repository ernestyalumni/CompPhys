/**
 * @file   : constant_eg1.cu
 * @brief  : Examples of using constant memory for CUDA, with smart pointers 
 * @details : constant memory for CUDA examples
 *  
 * @author : Ernest Yeung <ernestyalumni@gmail.com>
 * @date   : 20170103      
 * @ref    : http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-specifiers
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
 * nvcc constant_eg.cu -o constant_eg
 * 
 * */
#include <iostream>  
#include <memory> // std::unique_ptr
#include <array>  // std::array
#include <math.h> // std::exp

/*
// custom deleter as lambda function for float arrays (RR=real numbers=floats)
auto del_RRarr_lambda=[&](float* ptr) { cudaFree(ptr); }; 
// custom deleter as lambda function for int arrays (ZZ=integers=ints)
auto del_ZZarr_lambda=[&](int* ptr) { cudaFree(ptr); }; 

/*  error: expected a type specifier // constant_eg1.cu(39): error: variable "del_RRarr_lambda" is not a type name
 * obtained when trying to initialize in type declaration in struct */
/*
struct S1_unique_lambda { 
	// (data) members 
//	std::unique_ptr<float[],decltype(del_RRarr_lambda)> dev_X_uptr(nullptr, del_RRarr_lambda);
//	std::unique_ptr<int[], decltype(del_ZZarr_lambda)> dev_S_uptr(nullptr, del_ZZarr_lambda); 
//	std::unique_ptr<float[],decltype(del_RRarr_lambda)> dev_X_uptr; 
//	std::unique_ptr<int[], decltype(del_ZZarr_lambda)> dev_S_uptr;  
//	std::unique_ptr<float[]> dev_X_uptr; // error: 
	/* error: no operator "=" matches these operands
            operand types are: std::unique_ptr<float [], std::default_delete<float []>> = std::unique_ptr<float [], lambda [](float *)->void>
	*/
	/*
	std::unique_ptr<int[]> dev_S_uptr;  


	size_t Lx; 	// 8 bytes
	size_t Ly;  // 8 bytes
	unsigned long long Nx;	// 8 bytes
	unsigned long long Ny;  // 8 bytes

	// constructor
	S1_unique_lambda(size_t Lx,size_t Ly,unsigned long long Nx,unsigned long long Ny); 

	// move constructor
	S1_unique_lambda(S1_unique_lambda &&);  
	
	// operator overload assignment = 
	S1_unique_lambda &operator=(S1_unique_lambda &&); 
};

// constructor 
S1_unique_lambda::S1_unique_lambda(size_t Lx,size_t Ly,unsigned long long Nx,unsigned long long Ny) : 
	Lx {Lx}, Ly {Ly}, Nx {Nx}, Ny{Ny} 
{ 
	std::unique_ptr<float[],decltype(del_RRarr_lambda)> dev_X_uptr_new(nullptr,del_RRarr_lambda); 
	dev_X_uptr = std::move(dev_X_uptr_new);
	std::unique_ptr<int[],decltype(del_ZZarr_lambda)> dev_S_uptr_new(nullptr,del_ZZarr_lambda);
	dev_S_uptr = std::move(dev_S_uptr_new); 

};

// move constructor 
S1_unique_lambda::S1_unique_lambda(S1_unique_lambda&& old_struct) : 
	Lx { old_struct.Lx }, Ly { old_struct.Ly}, Nx { old_struct.Nx }, Ny { old_struct.Ny }, 
	dev_X_uptr{std::move(old_struct.dev_X_uptr) }, dev_S_uptr{std::move(old_struct.dev_S_uptr) } {};

// operator overload assignment = 
S1_unique_lambda & S1_unique_lambda::operator=(S1_unique_lambda && old_struct) {
	Lx = old_struct.Lx; 
	Ly = old_struct.Ly;
	Nx = old_struct.Nx;
	Ny = old_struct.Ny; 
	
	// unique_ptrs moved
	dev_X_uptr = std::move( old_struct.dev_X_uptr) ;
	dev_S_uptr = std::move( old_struct.dev_S_uptr); 
	
	return *this;
};
*/

// custom deleter as struct for float arrays (RR=real numbers=floats)
struct del_RRarr_struct { 
	void operator()(float* ptr) { cudaFree(ptr); } 
};  
// custom deleter as struct for int arrays (ZZ=integers=ints)
struct del_ZZarr_struct { 
	void operator()(int* ptr) { cudaFree(ptr); }
};

struct S1_unique_struct { 
	// (data) members , no dynamic initialization, but then no suitable constructor 
//	std::unique_ptr<float[],del_RRarr_struct> dev_X_uptr; //(nullptr, del_RRarr_struct());
//	std::unique_ptr<int[], del_ZZarr_struct> dev_S_uptr; // (nullptr, del_ZZarr_struct()); 
	std::array<float, 17> transProb; 

	size_t Lx;		// 8 bytes
	size_t Ly; 		// 8 bytes
	unsigned long long Nx;	// 8 bytes
	unsigned long long Ny;  // 8 bytes


/* no dynamic initialization
	// default constructor, needed by __constant__
	S1_unique_struct(); 

	// constructor 
	S1_unique_struct(size_t Lx, size_t Ly,unsigned long long Nx, unsigned long long Ny); 
	
	// move constructor
	S1_unique_struct(S1_unique_struct &&);  
	
	// operator overload assignment = 
	S1_unique_struct &operator=(S1_unique_struct &&); 
*/
};


/* no dynamic initialization 
// default constructor
S1_unique_struct::S1_unique_struct() {
/*	std::unique_ptr<float[],del_RRarr_struct> dev_X_uptr_new(nullptr,del_RRarr_struct()); 
	dev_X_uptr = std::move(dev_X_uptr_new);
	std::unique_ptr<int[],del_ZZarr_struct> dev_S_uptr_new(nullptr,del_ZZarr_struct());
	dev_S_uptr = std::move(dev_S_uptr_new); 	
}; 

/* 
 * error: dynamic initialization is not supported for __device__, __constant__ and __shared__ variables.
 */
	
// constructor 
/*
S1_unique_struct::S1_unique_struct(size_t Lx,size_t Ly,unsigned long long Nx,unsigned long long Ny) : 
	Lx {Lx}, Ly {Ly}, Nx {Nx}, Ny{Ny} 
{ 
	std::unique_ptr<float[],del_RRarr_struct> dev_X_uptr_new(nullptr,del_RRarr_struct()); 
	dev_X_uptr = std::move(dev_X_uptr_new);
	std::unique_ptr<int[],del_ZZarr_struct> dev_S_uptr_new(nullptr,del_ZZarr_struct());
	dev_S_uptr = std::move(dev_S_uptr_new); 

};

// move constructor 
S1_unique_struct::S1_unique_struct(S1_unique_struct&& old_struct) : 
	Lx { old_struct.Lx }, Ly { old_struct.Ly}, Nx { old_struct.Nx }, Ny { old_struct.Ny }, 
	dev_X_uptr{std::move(old_struct.dev_X_uptr) }, dev_S_uptr{std::move(old_struct.dev_S_uptr) } {};

// operator overload assignment = 
S1_unique_struct & S1_unique_struct::operator=(S1_unique_struct && old_struct) {
	Lx = old_struct.Lx; 
	Ly = old_struct.Ly;
	Nx = old_struct.Nx;
	Ny = old_struct.Ny; 
	
	// unique_ptrs moved
	dev_X_uptr = std::move( old_struct.dev_X_uptr) ;
	dev_S_uptr = std::move( old_struct.dev_S_uptr); 
	
	return *this;
};
*/
__constant__ S1_unique_struct constS1_uniq_struct;  



int main(int argc, char* argv[]) {
//	std::cout << " sizeof S1_unique_lambda : " << sizeof(S1_unique_lambda) << std::endl; 
	std::cout << " sizeof S1_unique_struct : " << sizeof(S1_unique_struct) << std::endl; 

	/* "boilerplate" test values */
	// on host
	std::array<float,17> h_transProb; 
	for (int i=0; i<17; i++) {
		h_transProb[i] = std::exp( (float) i) ;
	}

	S1_unique_struct s1_uniq_struct { h_transProb, 256,128,64,32 }; 

	cudaMemcpyToSymbol(constS1_uniq_struct, &s1_uniq_struct, sizeof(S1_unique_struct)); 

// this works as well
//	cudaMemcpyToSymbol(constS1_uniq_struct, &s1_uniq_struct, sizeof(s1_uniq_struct)); 
	
	/* sanity check */
	S1_unique_struct h_s1_uniq_struct; 
	cudaMemcpyFromSymbol(&h_s1_uniq_struct, constS1_uniq_struct, sizeof(S1_unique_struct)); 
	std::cout << " Lx : " << h_s1_uniq_struct.Lx << " Ly : " << h_s1_uniq_struct.Ly <<  
		" Nx : " << h_s1_uniq_struct.Nx << " Ny : " << h_s1_uniq_struct.Ny << std::endl; 
	for (int i =0; i<17; i++) {
		std::cout << h_s1_uniq_struct.transProb[i]	<< " ";
	}


}
