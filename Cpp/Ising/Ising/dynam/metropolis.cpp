/**
 * @file   : metropolis.cpp
 * @brief  : Metropolis algorithm for 2-dim. grid, with initialization, separate/implementation file, in C++11/14, 
 * @details : initialize function, ran1, metropolis function
 * 
 * @author : Ernest Yeung <ernestyalumni@gmail.com>
 * @date   : 20171229    
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
#include "./metropolis.h"  

double ran1(long *idum) 
{
	constexpr const int IA {16807};
	constexpr const int IM {2147483647}; 
	constexpr const float AM { 1.0/IM }; 
	constexpr const int IQ {127773}; 
	constexpr const int IR {2836}; 
	constexpr const int NTAB {32}; 
	constexpr const int NDIV { (1+(IM-1)/NTAB ) } ; 
	constexpr const double EPS {1.2e-7}; 
	constexpr const double RNMX (1.0-EPS); 
	
	int 		j;
	long 		k;
	static long iy = 0;
	static long iv[NTAB]; 
	double 		temp;
	
	if (*idum <=0 || !iy) {
		if (-(*idum) < 1) { 
			*idum = 1; }
		else { *idum = -(*idum); }
		for (j = NTAB + 7; j >=0 ; j--) {
			k = (*idum)/IQ;
			*idum = IA*(*idum - k*IQ) - IR*k; 
			if (*idum <0) { 
				*idum += IM; }
			if (j <NTAB) { 
				iv[j] = *idum; 
			}
		} // END for loop for j 
		iy = iv[0]; 
	}
	k  		= (*idum)/IQ;
	*idum 	= IA*(*idum -k*IQ)-IR*k;
	if (*idum <0) { *idum += IM; }
	j 	= iy/NDIV;
	iy 	= iv[j]; 
	iv[j] = *idum;
	if ((temp=AM*iy) > RNMX) { return RNMX ; } 
	else { return temp; } 
};  

/**
 * @fn initialize
 * @brief function to initialize energy, spin matrix, and magnetization 
 * */
void initialize(Spins2d& spins2d, Sysparam& sysparams) 
{ 
	size_t Lx = spins2d.L_is[0]; 
	size_t Ly = spins2d.L_is[1]; 
	
	double E = sysparams.E; 
	double M = sysparams.M; 

	// setup spin matrix and initial magnetization
	for (int j=0; j< Ly; j++) { 
		for (int i=0; i <Lx; i++) { 
			int k = i + Lx * j; 
			spins2d.S[k] = 1; // spin orientation for the ground state
			M += (double) spins2d.entry(i,j); 
		}
	}
	sysparams.M = M;

	// setup initial energy
	for (int j=0; j<Ly; j++) {
		for (int i=0; i<Lx; i++) {
			E -= (double) spins2d.entry(i,j) * 
					( spins2d.entry( periodic(i,Lx,-1), j) + 
					  spins2d.entry( i, periodic(j,Ly, -1) ) );
		} 
	}
	sysparams.E = E;
} // end of function initialize 


/**
 * @fn Metropolis
 * @param idum - long idum is random long to seed random numbers for ran1 
 * */
void Metropolis(long & idum, Spins2d& spins2d, Sysparam & sysparams, TransProb& transprob, Avg& avg) 
{ 
	size_t Lx = spins2d.L_is[0]; 
	size_t Ly = spins2d.L_is[1]; 
	
	double E = sysparams.E; 
	double M = sysparams.M; 

	
	// loop over all spins on the 2-dim. grid, pick a random spin (site) each time 
	for (size_t j = 0; j < Ly; j++) { 
		for (size_t i =0; i< Lx; i++) { 
			// pick random spin
			int ix = (int) (ran1(&idum)*(double)Lx); 
			int iy = (int) (ran1(&idum)*(double)Ly); 
			
			// essentially a stencil operation
			int DeltaE = 2* spins2d.entry(ix,iy) * 
				( spins2d.entry( periodic(ix,Lx,-1), iy) + spins2d.entry( periodic(ix,Lx,1), iy) + 
				  spins2d.entry( ix, periodic(iy,Ly,-1)) + spins2d.entry( ix, periodic(iy,Ly,1)) 
				 );
			
			// roll dice, see if we transition or not, given transprob
			if (ran1(&idum) <= transprob.get_by_DeltaE(DeltaE) ) { 
				int k = i + Lx*j; 
				spins2d.S[k] *= -1; // flip 1 spin and accept new spin config 
				
				M += (double) 2*spins2d.entry(i,j); 
				E += (double) DeltaE; 
			}
		} // END of for loop i 
	} // END of for loop j 
	sysparams.E = E;
	sysparams.M = M; 

	// update expectation values
	avg.Eavg += E; 
	avg.Mavg += M; 
	avg.Esq_avg += E*E; 
	avg.Msq_avg += M*M; 
	avg.absM_avg += std::fabs(M); 
		
} 
 

