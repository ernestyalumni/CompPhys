/** 
 * @file : binary_vals.cpp
 * 
 * cf. http://en.cppreference.com/w/cpp/utility/bitset/bitset
 * 
 * I tested compilation with g++, 
 * g++ -Wall -o "binary_vals" "binary_vals.cpp" 
 * and nvcc,
 * nvcc -std=c++11 binary_vals.cpp -o binary_vals_nv.exe
 * 
 * */
#include <iostream>
#include <bitset>  

int main() {
	std::bitset<6> x;  // 000000

	std::bitset<6> C_B = 0x0000 ; 

	std::bitset<6> B_N = 0x0001 ; 
	std::bitset<6> B_S = 0x0002 ; 
	std::bitset<6> B_W = 0x0004 ; 
	std::bitset<6> B_O = 0x0008 ; 
	std::bitset<6> B_NW = 0x0005 ; 
	std::bitset<6> B_SW = 0x0006 ; 
	std::bitset<6> B_NO = 0x0009 ; 
	std::bitset<6> B_SO = 0x000a ; 

	std::bitset<6> C_F = 0x0010 ; 

	std::bitset<6> C_E = 0x1000 ; 
	std::bitset<6> C_N = 0x0800 ; 
	std::bitset<6> C_S = 0x0400 ; 
	std::bitset<6> C_W = 0x0200 ; 

	
	int C_B_int = 0x0000 ;
	int B_N_int = 0x0001 ;
	int B_S_int = 0x0002 ; 
	int B_W_int = 0x0004 ; 
	int B_O_int = 0x0008 ; 

	int B_NW_int = 0x0005 ;
	int B_SW_int = 0x0006 ; 
	int B_NO_int = 0x0009 ; 
	int B_SO_int = 0x000a ; 

	
	int C_F_int = 0x0010 ;

	int C_E_int = 0x1000 ;
	int C_N_int = 0x0800 ; 
	int C_S_int = 0x0400 ; 
	int C_W_int = 0x0200 ; 

	int RHS_int = 0x0100 ; // from uvp.c, COMP_RHS

	
	std::cout << x << std::endl;
	std::cout << C_B << std::endl;

	std::cout << B_N << std::endl;
	std::cout << B_S << std::endl;
	std::cout << B_W << std::endl;
	std::cout << B_O << std::endl;
	std::cout << B_NW << std::endl;
	std::cout << B_SW << std::endl;
	std::cout << B_NO << std::endl;
	std::cout << B_SO << std::endl;

	std::cout << C_F << std::endl;

	std::cout << C_E << std::endl;
	std::cout << C_N << std::endl;
	std::cout << C_S << std::endl;
	std::cout << C_W << std::endl;

	std::cout << C_B_int << std::endl;

	std::cout << B_N_int << std::endl;
	std::cout << B_S_int << std::endl;
	std::cout << B_W_int << std::endl;
	std::cout << B_O_int << std::endl;
	
	std::cout << C_F_int << std::endl;

	std::cout << C_E_int << std::endl;
	std::cout << C_N_int << std::endl;
	std::cout << C_S_int << std::endl;
	std::cout << C_W_int << std::endl;

	std::cout << RHS_int << std::endl;

	std::cout << std::bitset<6>(C_B_int) << std::endl;

	std::cout << std::bitset<6>(B_N_int) << std::endl;
	std::cout << std::bitset<6>(B_S_int) << std::endl;
	std::cout << std::bitset<6>(B_W_int) << std::endl;
	std::cout << std::bitset<6>(B_O_int) << std::endl;


	std::cout << B_NW_int << std::endl;
	std::cout << B_SW_int << std::endl;
	std::cout << B_NO_int << std::endl;
	std::cout << B_SO_int << std::endl;

	std::cout << std::bitset<6>(B_NW_int) << std::endl;
	std::cout << std::bitset<6>(B_SW_int) << std::endl;
	std::cout << std::bitset<6>(B_NO_int) << std::endl;
	std::cout << std::bitset<6>(B_SO_int) << std::endl;

	std::cout << (B_N_int & 0x000f) << std::endl;
	std::cout << (B_S_int & 0x000f) << std::endl;
	std::cout << (B_W_int & 0x000f) << std::endl;
	std::cout << (B_O_int & 0x000f) << std::endl;

	bool eps_test_bool = !(C_E_int < C_F_int)  ;
	int eps_test_int = ((int) !(C_E_int < C_F_int) );
	float eps_test_float = ((float) !(C_E_int < C_F_int) ) ;

	std::cout << "Print out boolean values in different type casts " << std::endl;
	std::cout << eps_test_bool << std::endl ; 
	std::cout << eps_test_int << std::endl ; 
	std::cout << eps_test_float << std::endl ; 

	return 0;
}
