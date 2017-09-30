/**
 * @file   : 036sgemm_unified.cu
 * @brief  : C = alpha op(A) op(B) + beta C
 * @author : Ernest Yeung <ernestyalumni@gmail.com>
 * @date   : 20170928  
 * @ref    : https://developer.nvidia.com/sites/default/files/akamai/cuda/files/Misc/mygpu.pdf
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
 * nvcc -std=c++11 -lcublas 036sgemm.cu -o 036sgemm.exe
 * 
 * */
#include <iostream>
#include "cublas_v2.h"  

#include <memory>
#include <vector>

struct CuHandle { 
	cublasHandle_t handle;
	cublasStatus_t stat; 

	// constructor
	CuHandle() {
		stat = cublasCreate(&handle);			// initialize CUBLAS context
	}
	~CuHandle() { cublasDestroy(handle); }

	cublasHandle_t getHandle() {
		return handle;
	}
};
	

int main(int argc, char* argv[]) {
	constexpr const int m = 6; 			// a - mxk matrix
	constexpr const int n = 4;			// b - kxn matrix
	constexpr const int k = 5; 			// c - mxn matrix

	cudaError_t cudaStat;				// cudaMalloc status 
	cublasStatus_t stat;				// CUBLAS functions status

	// CUBLAS is column-major ordering 
	std::vector<float> a(m*k,0.f);				// mxk matrix a on the host
	std::vector<float> b(k*n,0.f);				// kxn matrix a on the host
	std::vector<float> c(m*n,0.f);				// mxn matrix a on the host


	// define a mxk matrix a column by column
	int ind=11;								// a:
	for (int j=0;j<k;j++) {						// 11,17,23,29,35
		for (int i=0;i<m;i++) {					// 12,18,24,30,36
			a[ j*m + i] = (float) ind++;	// 13,19,25,31,37
		}									// 14,20,26,32,38
	}										// 15,21,27,33,39
											// 16,22,28,34,40

	// define a kxn matrix b column by column
	ind=11;								// b:
	for (int j=0;j<n;j++) {						// 11,16,21,26
		for (int i=0;i<k;i++) {					// 12,17,22,27
			b[ j*k + i] = (float) ind++;	// 13,18,23,28
		}									// 14,19,24,29
	}										// 15,20,25,30

	// define a mxn matrix c column by column
	ind=11;								// c:
	for (int j=0;j<n;j++) {						// 11,17,23,29
		for (int i=0;i<m;i++) {					// 12,18,24,30
			c[ j*m + i] = (float) ind++;	// 13,19,25,31
		}									// 14,20,26,32
	}										// 15,21,27,33
											// 16,22,28,34

	for (auto ele : a) { std::cout << ele ; }; std::cout << std::endl; 
	for (auto ele : b) { std::cout << ele ;  }; std::cout << std::endl;
	for (auto ele : c) { std::cout << ele ; }; std::cout << std::endl; 
	
	// CUBLAS context; 
//	auto deleter_Handle=[&](cublasHandle_t* ptr){ cublasDestroy(*ptr); };

	// Segmentation Fault
//	std::shared_ptr<cublasHandle_t> handle_sh(new cublasHandle_t, deleter_Handle);
	
//	auto handle_sh = std::make_shared<cublasHandle_t>(new cublasHandle_t, deleter_Handle);

//	std::unique_ptr<cublasHandle_t, decltype(deleter_Handle)> handle_un(
//		new cublasHandle_t, deleter_Handle);
	cublasHandle_t handle;
	stat = cublasCreate(&handle);			// initialize CUBLAS context

//	auto deleter_Handle=[&](cublasHandle_t* ptr){ cublasDestroy(*ptr); };
//	std::shared_ptr<cublasHandle_t> handle_sh(new cublasHandle_t, deleter_Handle);
//	std::unique_ptr<cublasHandle_t, decltype(deleter_Handle)> handle_un(
//		new cublasHandle_t, deleter_Handle);
//	void DestroyHandle(cublasHandle_t* handle) {
//		cublasDestroy(*handle); }

	// works
//	std::unique_ptr<cublasHandle_t> handle_un(new cublasHandle_t); 	// works 

//	auto constructor_Handle=[&](cublasHandle_t* ptr){ cublasCreate(ptr); };

//	std::unique_ptr<cublasHandle_t,decltype(deleter_Handle)> handle_un(
//		c, deleter_Handle); 	// works 

	CuHandle cuHandle; 
	
	// Allocate problem device arrays
	auto deleter_mat=[&](float* ptr){ cudaFree(ptr); };
	std::shared_ptr<float> d_a(new float[m*k], deleter_mat);	// mxk matrix a on the device
	cudaMallocManaged((void **)&d_a, m*k*sizeof(float)); 			// device memeory for a
	std::shared_ptr<float> d_b(new float[k*n], deleter_mat);	// kxn matrix b on the device
	cudaMallocManaged((void **)&d_b, k*n*sizeof(float)); 			// device memeory for b
	std::shared_ptr<float> d_c(new float[m*n], deleter_mat);	// mxn matrix c on the device
	cudaMallocManaged((void **)&d_c, m*n*sizeof(float)); 			// device memeory for c

	std::unique_ptr<float[],decltype(deleter_mat)> d_a_u(
		new float[m*k], deleter_mat);	// mxk matrix a on the device
	cudaMallocManaged((void **)&d_a_u, m*k*sizeof(float)); 			// device memeory for a
	std::unique_ptr<float[],decltype(deleter_mat)> d_b_u(
		new float[k*n], deleter_mat);	// kxn matrix a on the device
	cudaMallocManaged((void **)&d_b_u, m*k*sizeof(float)); 			// device memeory for a
	std::unique_ptr<float[],decltype(deleter_mat)> d_c_u(
		new float[m*n], deleter_mat);	// mxk matrix a on the device
	cudaMallocManaged((void **)&d_c_u, m*n*sizeof(float)); 			// device memeory for a
	

	float a1 = 1.0f; 		// a1=1
	float bet = 1.0f;		// bet=1
	
	stat = cublasSetMatrix(m,k,sizeof(float),a.data(),m,d_a.get(),m); 	// a -> d_a
	stat = cublasSetMatrix(k,n,sizeof(float),b.data(),k,d_b.get(),k); 	// b -> d_b
	stat = cublasSetMatrix(m,n,sizeof(float),c.data(),m,d_c.get(),m); 	// c -> d_c

	stat = cublasSetMatrix(m,k,sizeof(float),a.data(),m,d_a_u.get(),m); 	// a -> d_a
	stat = cublasSetMatrix(k,n,sizeof(float),b.data(),k,d_b_u.get(),k); 	// b -> d_b
	stat = cublasSetMatrix(m,n,sizeof(float),c.data(),m,d_c_u.get(),m); 	// c -> d_c


	// matrix-matrix multiplication: d_c = a1*d_a*d_b + bet*d_c
	// d_a - mxk matrix, d_b -kxn matrix, d_c -mxn matrix;
	// a1,bet -scalars

	stat=cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,&a1,d_a.get(),
		m,d_b.get(),k,&bet,d_c.get(),m);  

	stat=cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,&a1,d_a_u.get(),
		m,d_b_u.get(),k,&bet,d_c_u.get(),m);  


//	stat=cublasSgemm(*(handle_un.get()),CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,&a1,d_a_u.get(),
//		m,d_b_u.get(),k,&bet,d_c_u.get(),m);  

	stat=cublasSgemm(cuHandle.getHandle(),CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,&a1,d_a_u.get(),
		m,d_b_u.get(),k,&bet,d_c_u.get(),m);  

		
	cudaDeviceSynchronize();
	
	std::cout << "c after Sgemm : " << std::endl;  
	for (int i=0;i<m;i++) {
		for (int j=0;j<n;j++) {
			std::cout << d_c.get()[j*m+i] << " "; // print c after Sgemm
		}
		std::cout << std::endl;
	}

	std::cout << "(unique ptr) c after Sgemm : " << std::endl;  
	for (int i=0;i<m;i++) {
		for (int j=0;j<n;j++) {
			std::cout << d_c_u.get()[j*m+i] << " "; // print c after Sgemm
		}
		std::cout << std::endl;
	}


	cublasDestroy(handle);				// destroy CUBLAS context

//	cublasDestroy(handle_un);

} 
 
