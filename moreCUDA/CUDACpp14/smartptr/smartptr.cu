/**
 * @file   : smartptr.cu
 * @brief  : Smart pointers content/source file in CUDA C++14, 
 * @author : Ernest Yeung <ernestyalumni@gmail.com>
 * @date   : 20171007  
 * @ref    :  
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
 * nvcc -std=c++14 -dc smartptr.cu -o smartptr.o
 * 
 * */
#include "smartptr.h"

/**
 * *** @name function (factories)
 * @note function factory : Lx \in \mathbb{Z}^+ |-> (\mapsto) u \in \mathbb{R}^{Lx} 
 * */


std::unique_ptr<float[], deleterRR_struct> make_uniq_u(const int Lx) {
	// inside function factory, custom deleter as stateless lambda function 
//	auto deleterRR_lambda=[&](float* ptr){ cudaFree(ptr); };

	std::unique_ptr<float[], deleterRR_struct> d_u(new float[Lx], deleterRR_struct()); 

	cudaMallocManaged((void **) &d_u, Lx*sizeof(float));
	return d_u;
}


/*
 * std::shared_ptr function (factory) and classes/structs (std::shared_ptr as a class member)
 * */
std::shared_ptr<float> make_sh_u(const int Lx) {
	// inside function factory, custom deleter as stateless lambda function 
	auto deleterRR_lambda=[&](float* ptr){ cudaFree(ptr); };

	std::shared_ptr<float> d_sh(new float[Lx], deleterRR_lambda); 
	cudaMallocManaged((void **) &d_sh, Lx*sizeof(float));
	return d_sh;
}
  
/*
 * *** END of function (factories) ***
 * */
 
 
/*
 * *** classes with smart pointers as member functions
 * */ 

/** 
 * 	@class RRModule
 * 	@ref	https://katyscode.wordpress.com/2012/10/04/c11-using-stdunique_ptr-as-a-class-member-initialization-move-semantics-and-custom-deleters/
 */
// constructor 
RRModule::RRModule(const int Lx) : Lx(Lx) {
	std::unique_ptr<float[], deleterRR> d_u(new float[Lx]);
	cudaMallocManaged((void **) &d_u,Lx*sizeof(float));
	this->X = std::move(d_u);
}

// member functions
void RRModule::load_from_hvec(std::vector<float>& h_X) {
	cudaMemcpy(this->X.get(), h_X.data(), Lx*sizeof(float),cudaMemcpyHostToDevice);	
}	

void RRModule::load_from_d_X(std::vector<float>& h_out) {
	cudaMemcpy(h_out.data(), this->X.get(), Lx*sizeof(float),cudaMemcpyDeviceToHost);
}		

/*
std::unique_ptr<float[], deleterRR> RRModule::get() {
	auto ptr=std::move(this->X);
	return ptr;
}
*/

// destructor
RRModule::~RRModule() {}


// constructor 
RRModule_sh::RRModule_sh(const int Lx) : Lx(Lx) {
	std::shared_ptr<float> d_sh(new float[Lx], deleterRR());
	cudaMallocManaged((void **) &d_sh,Lx*sizeof(float));
	this->X = std::move(d_sh);
}

// member functions
void RRModule_sh::load_from_hvec(std::vector<float>& h_X) {
	cudaMemcpy(this->X.get(), h_X.data(), Lx*sizeof(float),cudaMemcpyHostToDevice);	
}	

void RRModule_sh::load_from_d_X(std::vector<float>& h_out) {
	cudaMemcpy(h_out.data(), this->X.get(), Lx*sizeof(float),cudaMemcpyDeviceToHost);
}		

void RRModule_sh::load_from_uniq(std::unique_ptr<float[],deleterRR> & ptr_unique) {
	this->X = std::move(ptr_unique) ;
}


std::shared_ptr<float> RRModule_sh::get() {
//	return this->X;
	auto ptr = std::move(X);
	return ptr;
}


// destructor
RRModule_sh::~RRModule_sh() {}

