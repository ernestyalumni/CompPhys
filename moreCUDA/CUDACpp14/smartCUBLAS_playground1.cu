/**
 * @file   : smartCUBLAS_playground.cu
 * @brief  : Smart pointers (shared and unique ptrs) playground with CUBLAS, in C++14, 
 * @details : A playground to try out things in smart pointers with CUBLAS; 
 * 				especially abstracting our use of smart pointers with CUDA.  
 * 				Notice that std::make_unique DOES NOT have a custom deleter! (!!!)
 * 				Same with std::make_shared!  
 * 			cf. https://stackoverflow.com/questions/34243367/how-to-pass-deleter-to-make-shared
 * @author : Ernest Yeung <ernestyalumni@gmail.com>
 * @date   : 20170904  
 * @ref    : cf. https://www.codeproject.com/Tips/459832/Unique-ptr-custom-deleters-and-class-factories
 * 				Scott Meyers Effective Modern C++
 * 				http://shaharmike.com/cpp/unique-ptr/
 * 			https://katyscode.wordpress.com/2012/10/04/c11-using-stdunique_ptr-as-a-class-member-initialization-move-semantics-and-custom-deleters/
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
#include <vector> 	// std::vector

#include "cublas_v2.h" 

#include "smartptr/smartptr.h"

// custom deleter as a function for cublasHandle 
void del_cublasHandle_f(cublasHandle_t* ptr) { cublasDestroy(*ptr); };

/*
void Prod(const int m, const int n, const int k, 
		const float a1,
		std::unique_ptr<float[], deleterRR_struct>& A, 
		std::unique_ptr<float[], deleterRR_struct>& B, 
		const float bet,
		std::unique_ptr<float[], deleterRR_struct>& C) {
	
	auto del_cublasHandle=[&](cublasHandle_t* ptr) { cublasDestroy(*ptr); };

	std::unique_ptr<cublasHandle_t,decltype(del_cublasHandle)> handle_u(
		new cublasHandle_t, del_cublasHandle);
	cublasCreate(handle_u.get());
			
	cublasSgemm(*handle_u.get(),CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,&a1,A.get(),m,B.get(),k,&bet,C.get(),m);
		
			
}
*/


int main(int argc, char* argv[]) {
	constexpr const int m=6; 								// a - mxk matrix
	constexpr const int n=4;								// b - kxn matrix
	constexpr const int k=5;								// c - mxn matrix	

	// Notice that in CUBLAS, matrices follow COLUMN-major ordering

	// "boilerplate" values on host
		// define an mxk matrix a column by column
	std::vector<float> a(m*k);
	std::vector<float> b(k*n);
	std::vector<float> c(m*n);

	int i,j;								// i-row index,j-column index

	int ind=11;								// a:
	for (j=0;j<k;j++) { 					// 11,17,23,29,35 
		for (i=0;i<m;i++) { 				// 12,18,24,30,36
			a[i +j*m]=(float) ind++;			// 13,19,25,31,37
		}									// 14,20,26,32,38
	}										// 15,21,27,33,39
											// 16,22,28,34,40
	
	// print a row by row
	std::cout << " a: " << std::endl;
	for (i=0;i<m;i++) {
		for (j=0;j<k;j++) {
			std::cout << a[i+j*m] << " " ; }
		std::cout << std::endl; 
	}
	// define a kxn matrix b column by column
	ind=11;									// b:
	for (j=0;j<n;j++) {						// 11,16,21,26
		for (i=0;i<k;i++) {					// 12,17,22,27
			b[i+j*k]=(float) ind++;			// 13,18,23,28 
		}									// 14,19,24,29
	}										// 15,20,25,30
	// print b row by row
	std::cout << " b: " << std::endl;
	for (i=0;i<k;i++) {
		for (j=0;j<n;j++) {
			std::cout << b[i+j*k] << " " ; } 
		std::cout << std::endl;
	}
	
	// define an mxn matrix c column by column
	ind=11;									// c:
	for (j=0;j<n;j++) {						// 11,17,23,29
		for (i=0;i<m;i++) {					// 12,18,24,30
			c[i+j*m]=(float)ind++;			// 13,19,25,31
		}									// 14,20,26,32
	}										// 15,21,27,33
											// 16,22,28,34
	// print c row by row
	std::cout << "c: " << std::endl;
	for (i=0;i<m;i++){
		for (j=0;j<n;j++) {
			std::cout << c[i +j*m] << " ";
		}
		std::cout << std::endl;
	}
	
	auto A = make_uniq_u(m*k);
	auto B = make_uniq_u(k*n);
	auto C = make_uniq_u(m*n);

	// let's try cudaMemcpy instead of cublasSetMatrix to see what happens
	cudaMemcpy(A.get(), a.data(), sizeof(float)*m*k,cudaMemcpyHostToDevice);
	cudaMemcpy(B.get(), b.data(), sizeof(float)*k*n,cudaMemcpyHostToDevice);
	cudaMemcpy(C.get(), c.data(), sizeof(float)*m*n,cudaMemcpyHostToDevice);
	
	float a1=1.0f;
	float bet=1.0f;
	
	cudaDeviceSynchronize();
	cublasHandle_t handle;	// CUBLAS context
	cublasCreate(&handle);	// initialize CUBLAS context

	// custom deleter as a lambda function for cublasHandle 
	auto del_cublasHandle_lambda=[&](cublasHandle_t* ptr) { cublasDestroy(*ptr); };
	// custom deleter as a STRUCT for cublasHandle 
	struct del_cublasHandle_struct {
		void operator()(cublasHandle_t* ptr) { cublasDestroy(*ptr); }
	};

	/*
	 * shared_ptr for cublasHandle_t
	 * */
	std::shared_ptr<cublasHandle_t> handle_sh(new cublasHandle_t, del_cublasHandle_lambda);
	cublasCreate(handle_sh.get());

	std::shared_ptr<cublasHandle_t> handle_sh1(new cublasHandle_t, del_cublasHandle_f);
	cublasCreate(handle_sh1.get());

	std::shared_ptr<cublasHandle_t> handle_sh2(new cublasHandle_t, del_cublasHandle_struct());
	cublasCreate(handle_sh2.get());

	/*
	 * END of shared_ptr for cublasHandle_t
	 * */

	/*
	 * unique_ptr for cublasHandle_t
	 * */
/*	std::unique_ptr<cublasHandle_t,decltype(del_cublasHandle_lambda)> handle_u(
		new cublasHandle_t, del_cublasHandle_lambda);
	cublasCreate(handle_u.get());

	std::unique_ptr<cublasHandle_t,void(*)(cublasHandle_t*)> handle_u1(
		new cublasHandle_t, del_cublasHandle_f);
	cublasCreate(handle_u1.get());

	std::unique_ptr<cublasHandle_t,del_cublasHandle_struct> handle_u2(
		new cublasHandle_t, del_cublasHandle_struct());
	cublasCreate(handle_u2.get());
	
	// this WORKS; IGMORE->Segmentation fault missing custom deleter at initialization
	std::unique_ptr<cublasHandle_t,del_cublasHandle_struct> handle_u3(
		new cublasHandle_t);
	cublasCreate(handle_u3.get());
*/

	/*
	 * END of unique_ptr for cublasHandle_t
	 * */


	// this WORKS
	cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,&a1,A.get(),m,B.get(),k,&bet,C.get(),m);

	// this WORKS
//	cublasSgemm(*handle_sh1.get(),CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,&a1,A.get(),m,B.get(),k,&bet,C.get(),m);

	// this WORKS
//	cublasSgemm(*handle_sh2.get(),CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,&a1,A.get(),m,B.get(),k,&bet,C.get(),m);

	// this WORKS
//	cublasSgemm(*handle_u.get(),CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,&a1,A.get(),m,B.get(),k,&bet,C.get(),m);

	// this WORKS
//	cublasSgemm(*handle_u1.get(),CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,&a1,A.get(),m,B.get(),k,&bet,C.get(),m);

	// this WORKS
//	cublasSgemm(*handle_u2.get(),CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,&a1,A.get(),m,B.get(),k,&bet,C.get(),m);

	// this WORKS: IGNORE->this doesn't work; Segmentation fault missing custom deleter at initialization
//	cublasSgemm(*handle_u3.get(),CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,&a1,A.get(),m,B.get(),k,&bet,C.get(),m);

//	Prod(m,n,k,a1,A,B,bet,C);


	cudaDeviceSynchronize();

	std::vector<float> host_c(m*n);
	cublasGetMatrix(m,n,sizeof(float), C.get(),m,host_c.data(),m); // cp C -> host_c 
	
	std::cout << " c after Sgemm : " << std::endl; 
	for (i=0;i<m;i++) {
		for (j=0;j<n;j++){
			std::cout << host_c[i+j*m] << " "; 
		}
		std::cout << std::endl;
	}	
	
	
	// Clean up
	cublasDestroy(handle);	// destroy CUBLAS context
	cudaDeviceReset();
}
