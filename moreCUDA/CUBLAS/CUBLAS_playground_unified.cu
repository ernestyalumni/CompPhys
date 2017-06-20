/**	
 * @file   : CUBLAS_playground_unified.cu
 * @brief  : CUBLAS playground - trying out way to use CUBLAS, but with 
 * CUDA unified memory
 * @author : Ernest Yeung	ernestyalumni@gmail.com
 * @date   : 20170620
 * @ref    : 
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
 * P.S. I'm using an EVGA GeForce GTX 980 Ti which has at most compute capability 5.2; 
 * A hardware donation or financial contribution would help with obtaining a 1080 Ti with compute capability 6.2
 * so that I can do -arch='sm_62' and use many new CUDA Unified Memory features
 * */
 /**
  * COMPILATION TIP(s)
  * (without make file, stand-alone)
  * nvcc -std=c++11 -arch='sm_61' -lcublas CUBLAS_playground_unified.cu -o CUBLAS_playground_unified.exe
  * 
  * */
#include <iostream> 
#include "cublas_v2.h" 

// these have to be set manually
constexpr const int N_1 {5};
constexpr const int N_2 {4}; 

// matrix
__device__ __managed__ float A[N_1*N_2]; 

// vectors
__device__ __managed__ float x[N_2]; 
__device__ __managed__ float y[N_1]; 

__device__ __managed__ float E[N_1]; 
__device__ __managed__ float err_vec[N_1]; 

int main() {
	cublasStatus_t stat; 
	cublasHandle_t handle;
	cublasCreate(&handle);
	
	float a1=1.0f;
	float bet=0.0f;
	
	float a2 = 0.85;
	float a3 = (1.f - a2);
	float a4 = -1.f; 
	
	// boilerplate
	for (int i=0; i < N_2; ++i) {
		x[i] = ((float) i+2.f); 
	}
	
	for (int i=0; i < N_1; ++i) { 
		y[i] = ((float) (i+5.f)*0.1f ) ;
		E[i] = ((float) (i+7.f)*100.f) ; 
		err_vec[i] = ((float) (i+1.f)*0.001f ); 
	}
	for (int j =0; j<N_2;++j) { 
		for (int i =0; i<N_1;++i) {
			A[i + N_1*j] = ((float) (i + N_1*j)*10.f); 
		} 
	}
	
	std::cout << "Before operations : " << std::endl; 
	std::cout << " x : " << std::endl;

	for (int i=0; i < N_2; ++i) {
		std::cout << x[i] << " ";   
	}
	std::cout << std::endl;  

	std::cout << " y : " << std::endl;

	for (int i=0; i < N_1; ++i) {
		std::cout << y[i] << " ";  } 
	std::cout << std::endl ;	
		
	for (int i=0; i < N_1; ++i) {
		std::cout << E[i] << " ";  }
	std::cout << std::endl; 	
		
	for (int i =0; i<N_1;++i) {
		for (int j =0; j<N_2;++j) { 
			std::cout << A[i + N_1*j] << " ";
		} 
		std::cout << std::endl;  
	}
	
	// matrix-vector multiplication: y = A*x
	stat=cublasSgemv(handle,CUBLAS_OP_N,N_1,N_2,&a1,A,N_1,x,1,&bet,y,1);
	
	cudaDeviceSynchronize(); 
	std::cout << " After matrix-vector multiplication : " << std::endl; 
	for (int i=0; i < N_1; ++i) {
		std::cout << y[i] << " ";  } 
	std::cout << std::endl ;	
	
	// scale vector y
	
	cublasSscal(handle,N_1,&a2,y,1); 
	cudaDeviceSynchronize(); 
	std::cout << " After scaling, by : " << a2 << std::endl; 
	for (int i=0; i < N_1; ++i) {
		std::cout << y[i] << " ";  } 
	std::cout << std::endl ;	
	
	// add vector (1.0f - a2) * E i.e. 
	// y := (1.0f-alpha) * E + y
	cublasSaxpy(handle,N_1,&a3,E,1,y,1); 
	cudaDeviceSynchronize(); 
	std::cout << " After vector addition, with scaling by : " << a3 << std::endl; 
	for (int i=0; i < N_1; ++i) {
		std::cout << y[i] << " ";  } 
	std::cout << std::endl ;	
	
	// cublasSaxpy - compute y = alpha x + y 
	// cublasSaxpy(handle,n,&a1, x, 1, y, 1)
	
	// err_vec += 1.*y
	cublasSaxpy(handle,N_1,&a1, y,1, err_vec,1); 

	// err_vec += -1.*E  ==> err_vec = 1.*y - 1.*E
	cublasSaxpy(handle,N_1,&a4, E,1, err_vec,1); 
	cudaDeviceSynchronize(); 
	std::cout << " After vector addition (subtraction) : " << std::endl; 
	for (int i=0; i < N_1; ++i) {
		std::cout << err_vec[i] << " ";  } 
	std::cout << std::endl ;	
	 
	// cublasSasum - sum of absolute values
	float errsumresult = 0.f;
	stat=cublasSasum(handle,N_1,err_vec,1,&errsumresult); 
	cudaDeviceSynchronize(); 
	std::cout << " errsum result: " << errsumresult << " " << std::endl;  
	
	cudaMemset(err_vec,0.f,N_1*sizeof(float));
	
	stat=cublasSasum(handle,N_1,err_vec,1,&errsumresult); 
	cudaDeviceSynchronize(); 
	
	std::cout << " errsum result: " << errsumresult << " " << std::endl;  
	
	
}
