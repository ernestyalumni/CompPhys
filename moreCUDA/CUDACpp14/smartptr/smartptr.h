/**
 * @file   : smartptr.h
 * @brief  : Smart pointers header file in CUDA C++14, 
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
#ifndef __SMARTPTR_H__
#define __SMARTPTR_H__ 
 
#include <memory>  // std::shared_ptr, std::unique_ptr 
#include <vector>  // std::vector

/*
 * *** custom deleters *** 
 * */

// field K = float; RR = real numbers, float  
auto deleterRR_lambda=[&](float* ptr){ cudaFree(ptr); };

/* custom deleter as a struct */ 
struct deleterRR_struct
{
	void operator()(float* ptr) const 
	{
		cudaFree(ptr);
	}
};

/*
 * *** END of custom deleters ***
 * */

/**
 * *** @name function (factories)
 * @note function factory : Lx \in \mathbb{Z}^+ |-> (\mapsto) u \in \mathbb{R}^{Lx} 
 * */
 
//std::unique_ptr<float[], decltype(deleterRR_lambda)> make_uniq_u(const int); // undefined reference
std::unique_ptr<float[], deleterRR_struct> make_uniq_u(const int); // undefined reference
std::shared_ptr<float> make_sh_u(const int); 

/*
 * *** END of function (factories) ***
 * */

/*
 * *** classes with smart pointers as member functions
 * */ 

// RRmodule; RR = real numbers, float 
class RRModule
{
	private:
		int Lx; // remember you can use long

		// member custom deleter as struct; auto lambda not allowed here
		struct deleterRR {
			void operator()(float* ptr) const
			{
				cudaFree(ptr);
			}
		};

		
		// member
		std::unique_ptr<float[], deleterRR> X;

		
	public:
		// Constructor
		RRModule(const int);
		
		// member functions
		void load_from_hvec(std::vector<float>& );
		
		void load_from_d_X(std::vector<float>& );

		
		// destructor
		~RRModule();			
};

class RRModule_sh
{
	private:
		int Lx; // remember you can use long

		// member custom deleter as struct; auto lambda not allowed here
		struct deleterRR {
			void operator()(float* ptr) const
			{
				cudaFree(ptr);
			}
		};
		
		// member 
		std::shared_ptr<float> X;

		
	public:
		// Constructor
		RRModule_sh(const int);
		
		// member functions
		void load_from_hvec(std::vector<float>& );
		
		void load_from_d_X(std::vector<float>& );
		
		void load_from_uniq(std::unique_ptr<float[],deleterRR> &);
		
		std::shared_ptr<float> get();
		
		// destructor
		~RRModule_sh();			
};

#endif // __SMARTPTR_H__
