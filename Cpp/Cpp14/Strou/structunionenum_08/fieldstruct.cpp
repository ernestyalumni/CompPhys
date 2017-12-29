/**
 * @file   : fieldstruct.cpp
 * @brief  : struct of fields in C++111/14   
 * @details : struct of fields
 * 
 * @author : Ernest Yeung <ernestyalumni@gmail.com>
 * @date   : 20171229    
 * @ref    : Ch. 8 Structures, Unions, and Enumerations; Bjarne Stroustrup, The C++ Programming Language, 4th Ed.  
 * Addison-Wesley 
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
 * g++ fieldstruct.cpp -o fieldstruct
 * 
 * */

struct PPN { 	// R6000 Physical Page Number  
	unsigned int PFN : 22; 	// Page Frame Number
	int : 3;
	unsigned int CCA : 3;	// Cache Coherency Algorithm 
	bool nonreachable : 1;
	bool dirty : 1;
	bool valid : 1;
	bool global : 1;
};  

void part_of_VM_system(PPN* p)
{
	// ... 
	if (p->dirty) { 	// contents changed
		// copy to disk
		p->dirty =0;
	}
};  

int main(int argc, char* argv[]) {
	
}
