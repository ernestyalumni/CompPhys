/**
 * @file   : sysparam.cu
 * @brief  : Physical parameters of the system separate implementation file, in CUDA C++11/14, on device GPU 
 * @details : as a struct, have total energy E, total magnetization M, temperature of entire system T (in energy units)  
 * 
 * @author : Ernest Yeung <ernestyalumni@gmail.com>
 * @date   : 20180103    
 * @ref    : M. Hjorth-Jensen, Computational Physics, University of Oslo (2015)
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
 * g++ main.cpp ./structs/structs.cpp -o main
 * 
 * */
#include "./sysparam.h"

/* =============== device GPU structs =============== */ 

// getting functions
/** @fn get_by_DeltaE 
 * @details given DeltaE (\Delta E), DeltaE = -8J, -4J,...8J, we want to get the 
 * transition probability from std::unique_ptr transprob (but transprob indexed by 
 * 0,1,...(17-1)
 * */
float TransProb::get_by_DeltaE(const int DeltaE) {
		return transProb[DeltaE+8]; 
} 

  
/* struct of structs
 * I chose this design because 
 * 1. I want a struct such that its data members is identified by pseudo-mathematical symbols, 
 * i.e. a data member E should denote total Energy  
 * 1.a. I want this struct to be "empty" in that it has no custom constructors/destructors, 
 * so that it can be used by both host CPU and device GPU, and can be qualified with __constant__, so to be 
 * put in constant memory 
 * 2. I want a struct of these structs so to automate its construction/destruction: 
 * cudaMallocManaged, cudaFree, and use smart pointers for it
 * */

// default constructor
/** @fn Sysparam_ptr()
 * @brief default constructor for struct Sysparam_ptr 
 * @details set all E,M,T parameters to 0
 * */
Sysparam_ptr::Sysparam_ptr() { 
	std::unique_ptr<Sysparam, del_Sysparam_struct> d_sysparams_in(nullptr, del_Sysparam_struct() ); 
	cudaMallocManaged((void **) &d_sysparams_in, 1 * sizeof(Sysparam)) ;
	d_sysparams = std::move(d_sysparams_in);  
	
	d_sysparams->E = 0.f; d_sysparams->M = 0.f; d_sysparams->T = 0.f;}  

// constructors
/** @fn Sysparam_ptr(const float,const float,const float)
 * @brief constructor for struct Sysparam_ptr 
 * */
Sysparam_ptr::Sysparam_ptr(const float E, const float M, const float T) {  
	std::unique_ptr<Sysparam, del_Sysparam_struct> d_sysparams_in(nullptr, del_Sysparam_struct() ); 
	cudaMallocManaged((void **) &d_sysparams_in, 1 * sizeof(Sysparam)) ;
	d_sysparams = std::move(d_sysparams_in);  

	d_sysparams->E = E; d_sysparams->M = M; d_sysparams->T = T; }  

/** @fn Sysparam_ptr(const float)
 * @brief constructor for struct Sysparam_ptr, when only given the system temperature (initially)
 * */
Sysparam_ptr::Sysparam_ptr(const float T) { 
	std::unique_ptr<Sysparam, del_Sysparam_struct> d_sysparams_in(nullptr, del_Sysparam_struct() ); 
	cudaMallocManaged((void **) &d_sysparams_in, 1 * sizeof(Sysparam)) ;
	d_sysparams = std::move(d_sysparams_in);  

	d_sysparams->E = 0.f; d_sysparams->M = 0.f; d_sysparams->T = T; }  

// move constructor; necessitated by unique_ptr
Sysparam_ptr::Sysparam_ptr(Sysparam_ptr && old_sysparam_ptr) : 
	d_sysparams { std::move(old_sysparam_ptr.d_sysparams) }  {}     

// operator overload assignment = 
Sysparam_ptr & Sysparam_ptr::operator=(Sysparam_ptr && old_sysparam_ptr) {
	d_sysparams = std::move( old_sysparam_ptr.d_sysparams );
	return *this;
}



// default constructor
Avg_ptr::Avg_ptr() { 
	std::unique_ptr<Avg, del_Avg_struct> d_avgs_in(nullptr, del_Avg_struct() ); 
	cudaMallocManaged((void **) &d_avgs_in, 1 * sizeof(Avg)) ;
	d_avgs = std::move(d_avgs_in);  
	
	d_avgs->Eavg = 0.f; d_avgs->Mavg = 0.f; d_avgs->Esq_avg = 0.f; d_avgs->Msq_avg = 0.f; 
	d_avgs->absM_avg = 0.f; d_avgs->M4_avg = 0.f; 
}  

// constructors
Avg_ptr::Avg_ptr(const float Eavg, const float Mavg, const float Esq_avg, const float Msq_avg,
	const float absM_avg, const float M4_avg) { 
	std::unique_ptr<Avg, del_Avg_struct> d_avgs_in(nullptr, del_Avg_struct() ); 
	cudaMallocManaged((void **) &d_avgs_in, 1 * sizeof(Avg)) ;
	d_avgs = std::move(d_avgs_in);  
	
	d_avgs->Eavg = Eavg; d_avgs->Mavg = Mavg; d_avgs->Esq_avg = Esq_avg; d_avgs->Msq_avg = Msq_avg; 
	d_avgs->absM_avg = absM_avg; d_avgs->M4_avg = M4_avg; 
}  

// move constructor; necessitated by unique_ptr
Avg_ptr::Avg_ptr(Avg_ptr && old_avg_ptr) :  
	  d_avgs { std::move(old_avg_ptr.d_avgs) } {}

// operator overload assignment = 
Avg_ptr & Avg_ptr::operator=(Avg_ptr && old_avg_ptr)  {
	d_avgs = std::move( old_avg_ptr.d_avgs );
	return *this; 
}


// default constructor
TransProb_ptr::TransProb_ptr() { 
	std::unique_ptr<TransProb, del_TransProb_struct> d_transProb_in(nullptr, del_TransProb_struct() ); 
	cudaMallocManaged((void **) &d_transProb_in, 1 * sizeof(TransProb)) ;
	d_transProb = std::move(d_transProb_in);  
}	

// constructors
TransProb_ptr::TransProb_ptr(const float T, const float J) { 
	std::unique_ptr<TransProb, del_TransProb_struct> d_transProb_in(nullptr, del_TransProb_struct() ); 
	cudaMallocManaged((void **) &d_transProb_in, 1 * sizeof(TransProb)) ;
	d_transProb = std::move(d_transProb_in);  

	d_transProb->J = J; 
	for (int de = -8; de <= 8; de +=4) {
		(d_transProb->transProb)[de+8] = std::exp(-((float) de)/T); 
	}
	
//	cudaMemcpyToSymbol(constTransProb, &(this->d_transProb), sizeof(TransProb)*1); 
	
}	

// move constructor; necessitated by unique_ptr
TransProb_ptr::TransProb_ptr(TransProb_ptr && old_transProb_ptr) : 
	d_transProb { std::move(old_transProb_ptr.d_transProb ) }  {}   

// operator overload assignment = 
TransProb_ptr & TransProb_ptr::operator=(TransProb_ptr && old_transProb_ptr)  {
	d_transProb = std::move( old_transProb_ptr.d_transProb );
	return *this; 
}



/* ********** END of device GPU structs ********** */

