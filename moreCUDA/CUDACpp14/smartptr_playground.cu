/**
 * @file   : smartptr_playground.cu
 * @brief  : Smart pointers (shared and unique ptrs) playground, in C++14, 
 * @details : A playground to try out things in smart pointers; 
 * 				especially abstracting our use of smart pointers with CUDA.  
 * 				Notice that std::make_unique DOES NOT have a custom deleter! (!!!)
 * 				Same with std::make_shared!  
 * 			cf. https://stackoverflow.com/questions/34243367/how-to-pass-deleter-to-make-shared
 * @author : Ernest Yeung <ernestyalumni@gmail.com>
 * @date   : 20170904  
 * @ref    : cf. Scott Meyers Effective Modern C++
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
 * nvcc -std=c++14 smartptr_playground.cu -o smartptr_playground.exe
 * 
 * */
#include <iostream> // std::cout 
#include <memory>  // std::shared_ptr, std::unique_ptr 

#include <vector> 	// std::vector
#include <algorithm> // std::fill

// RR = real numbers, float, tells us what field K we're dealing with

/* custom deleter as stateless lambda function */
auto deleterRR_lambda=[&](float* ptr){ cudaFree(ptr); };

/* custom deleter as a FUNCTION */ 
void deleterRR(float* ptr) {
	cudaFree(ptr);
};

/* custom deleter as a struct */ 
struct deleterRR_struct
{
	void operator()(float* ptr) const 
	{
		cudaFree(ptr);
	}
};

// create unique_ptr instances with a "function factory", Lx \in \mathbb{Z}^+ |-> (\mapsto) u \in \mathbb{R}^{Lx}
/* explicit for explicit return type, the type in the beginning */
/* error : not a class or struct name 
 * no standard type for lambdas, only way to capture lambda with no conversion is auto 
 * cf. http://shaharmike.com/cpp/lambdas-and-functions/
 * 


std::unique_ptr<float[], (void)(float*)  > make_u_explicit(const int Lx) {
	// inside function factory, custom deleter as stateless lambda function 
	auto deleterRR_lambda=[&](float* ptr){ cudaFree(ptr); };

	std::unique_ptr<float[], decltype(deleterRR_lambda)> d_u(new float[Lx], deleterRR_lambda); 
	cudaMalloc((void **) &d_u, Lx*sizeof(float));
	return d_u;
};
*/

auto make_uniq_u(const int Lx) {
	// inside function factory, custom deleter as stateless lambda function 
	auto deleterRR_lambda=[&](float* ptr){ cudaFree(ptr); };

	std::unique_ptr<float[], decltype(deleterRR_lambda)> d_u(new float[Lx], deleterRR_lambda); 
	cudaMallocManaged((void **) &d_u, Lx*sizeof(float));
	return d_u;
};

/*
 * unique_ptr as class member?
 * cf. https://stackoverflow.com/questions/11302981/c11-declaring-non-static-data-members-as-auto
 * No you cannot for a class member, as decided by a committee
 * */
/*
class RRModule 
{
	private:
//	static const auto deleterRR_lambda=[&](float* ptr){ cudaFree(ptr); }; // lambda not allowed in constant expression
//	static auto deleterRR_lambda=[&](float* ptr){ cudaFree(ptr); };  // error: a lambda is not allowed in a constant expression
//		static const auto d_u; // initializer required
	public:
//		auto deleterRR_lambda=[&](float* ptr){ cudaFree(ptr); }; // auto not allowed here
//		static const auto d_u; // initializer required
		
};
*/
/*
struct RRModule {
	auto deleterRR_lambda=[&](float* ptr){ cudaFree(ptr); }; // auto not allowed here		
};*/

/** 
 * 	@class RRModule
 * 	@ref	https://katyscode.wordpress.com/2012/10/04/c11-using-stdunique_ptr-as-a-class-member-initialization-move-semantics-and-custom-deleters/
 */

class RRModule
{
	private:
		int Lx; // long type available as well
//		std::unique_ptr<float[], void (*)(float*)> u_u; // u_u = unique u \in RR^{Lx}
		// customer deleter as struct
		struct deleterRR {
			void operator()(float* ptr) const
			{
				cudaFree(ptr);
			}
		};
		std::unique_ptr<float[], deleterRR_struct> X;

	public:
		// constructor
		RRModule(const int Lx): Lx(Lx) {
			std::unique_ptr<float[], deleterRR_struct> d_u(new float[Lx]);
			cudaMallocManaged((void **) &d_u,Lx*sizeof(float));
//			X = d_u; // error here cannot be referenced -- it is a deleted function
			X = std::move(d_u);
		}
		
		void load_from_hvec(std::vector<float>& h_X) {
			cudaMemcpy(X.get(), h_X.data(), Lx*sizeof(float),cudaMemcpyHostToDevice);
			
		}	
		
		void load_from_d_X(std::vector<float>& h_out) {
			cudaMemcpy(h_out.data(), X.get(), Lx*sizeof(float),cudaMemcpyDeviceToHost);
		}			
};

/*
 * std::shared_ptr function (factory) and classes/structs (std::shared_ptr as a class member)
 * */
auto make_sh_u(const int Lx) {
	// inside function factory, custom deleter as stateless lambda function 
	auto deleterRR_lambda=[&](float* ptr){ cudaFree(ptr); };

	std::shared_ptr<float> d_sh(new float[Lx], deleterRR_lambda); 
	cudaMallocManaged((void **) &d_sh, Lx*sizeof(float));
	return d_sh;
};

class RRModule_sh
{
	private:
		int Lx; // long type available as well
		struct deleterRR {
			void operator()(float* ptr) const
			{
				cudaFree(ptr);
			}
		};
		std::shared_ptr<float> X;

	public:
		// constructor
		RRModule_sh(const int Lx): Lx(Lx) {
			std::shared_ptr<float> d_sh(new float[Lx],deleterRR());
			cudaMallocManaged((void **) &d_sh,Lx*sizeof(float));
			X = std::move(d_sh);
		}
		
		void load_from_hvec(std::vector<float>& h_X) {
			cudaMemcpy(X.get(), h_X.data(), Lx*sizeof(float),cudaMemcpyHostToDevice);
			
		}	
		
		void load_from_d_X(std::vector<float>& h_out) {
			cudaMemcpy(h_out.data(), X.get(), Lx*sizeof(float),cudaMemcpyDeviceToHost);
		}			
};



int main(int argc, char* argv[]) {
	constexpr const size_t Lx = {1<<5}; // 2^5 = 32

	/*
	 * std::unique_ptr on the device GPU
	 * */
	// device pointers
//	std::unique_ptr<float[], void (*)(float* )> d_u_in(new float[Lx], &deleterRR); /* return type has size of float[] 
//		 plus at least size of function pointer */
//	cudaMallocManaged((void **) &d_u_in, Lx*sizeof(float) +sizeof( &deleterRR ) ); // Segmentation Fault

	// Create unique_ptr with struct version of custom deleter

	// this WORKS
	std::unique_ptr<float[], deleterRR_struct> d_u_1  ; //(new float[Lx], deleterRR_struct);

	// this WORKS too, with the new constructor
	std::unique_ptr<float[], deleterRR_struct> d_u_2(new float[Lx] );
	// this WORKS
//	cudaMalloc((void **) &d_u_2, Lx*sizeof(float) );  
	cudaMallocManaged((void **) &d_u_2, Lx*sizeof(float) );  
		
	


	// Create a unique_ptr to an array of Lx floats
	std::unique_ptr<float[]> h_u = std::make_unique<float[]>(Lx); // h=host and u \in \mathbb{R}^{Lx}
	
	std::unique_ptr<float[]> d_u = std::make_unique<float[]>(Lx); // d=device and u \in \mathbb{R}^{Lx}
//	cudaMalloc((void **) &d_u, Lx*sizeof(float) ); // Segmentation Fault alone
	
	std::unique_ptr<float[], decltype(deleterRR_lambda)> d_u_d(new float[Lx], deleterRR_lambda);
	cudaMalloc((void **) &d_u_d, Lx*sizeof(float));

	std::unique_ptr<float[], decltype(deleterRR_lambda)> d_u_u(new float[Lx], deleterRR_lambda); // u for unique and unified
	cudaMallocManaged((void **) &d_u_u, Lx*sizeof(float));

	auto u_instance = make_uniq_u( Lx) ; 

	RRModule R( Lx);

	// Allocate host arrays
	std::vector<float> h_vec(Lx,1.f);
	std::fill(h_vec.begin()+1,h_vec.begin() + h_vec.size()/4, 3.f);
	std::fill(h_vec.begin()+h_vec.size()/4,h_vec.end() - h_vec.size()/5, 8.f);
	
	R.load_from_hvec(h_vec);
	
	// Readout result into output array on host
	std::vector<float> out_vec(Lx,0.f);
	R.load_from_d_X(out_vec);
	std::cout << " \n for unique_ptr on device : " << std::endl;
	for (auto ele : out_vec) {
		std::cout << " " << ele ; 
	}
	std::cout << std::endl;
	
	/*
	 * std::shared_ptr on the device GPU
	 * std::shared_ptr has twice the memory size as a raw pointer, but 
	 * let's explore if we cudaMalloc for only 1 raw pointer size!  
	 * */
	std::shared_ptr<float> d_sh(new float[Lx], deleterRR_lambda);
	// this WORKS
	// cudaMalloc((void **) &d_sh,Lx * sizeof(float));
	cudaMallocManaged((void**)&d_sh,Lx*sizeof(float));
	
	std::shared_ptr<float> d_sh_1(new float[Lx], deleterRR_struct());
	// this WORKS
	// cudaMalloc((void **) &d_sh_1,Lx * sizeof(float));
	cudaMallocManaged((void**)&d_sh_1,Lx*sizeof(float));
	
	auto sh_instance = make_sh_u(Lx);

	// shared version of RRModule
	RRModule_sh R_sh( Lx);

	R_sh.load_from_hvec(h_vec);
	R_sh.load_from_d_X(out_vec);
	std::cout << " \n for shared_ptr on device : " << std::endl;
	for (auto ele : out_vec) {
		std::cout << " " << ele ; 
	}
	std::cout << std::endl;
	
	
	// Clean up 
//	cudaFree( &d_u); // Segmentation Fault 
//	cudaFree( d_u.get() ); // Segmentation Fault

	cudaDeviceReset();
	return 0;
}
