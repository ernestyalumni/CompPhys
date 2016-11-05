/* scans.cu
 * Ernest Yeung
 * ernestyalumni@gmail.com
 * Demonstrates Hillis/Steele and Blelloch (exclusive) scan with a parallel implementation
 * with CUDA C/C++ and global memory
 * 
 * */
#include "scans.h" 

// parallel implementations

	// Blelloch scan, 1st part, reduce part or up sweep part, using global memory
	// k is for the kth iteration, equal to 2^k offset; L_x is length of input array, i.e. L_x = |f_in|
__global__ void Blelloch_up_global( float* f_in, float* f_out, const int k, const int L_x) {

	const int k_x = threadIdx.x + blockIdx.x * blockDim.x ;
	const int offset = 1 << k; // offset = 2^k
	
	// check if global thread index happens to fall out of global length of desired, target, input array
	if (k_x >= L_x) {
		return ; }

	float tempval = 0.f;
	// k_x = 2^kj-1, j \in \lbrace 1,2,\dots \lfloor N/2^k \rfloor \rbrace check
	if ( ((k_x%offset)==(offset-1)) && (k_x >= (offset - 1)) ) { 
		tempval = f_in[k_x] + f_in[k_x-offset/2]; }
	else {
		tempval = f_in[k_x] ; }
	f_out[k_x] = tempval;
}


__global__ void Blelloch_down_global( float* f_in, float* f_out, const int k, const int L_x) {

	const int k_x = threadIdx.x + blockIdx.x * blockDim.x ;
	const int offset { 1 << k }; // offset = 2^k
	
	// check if global thread index happens to fall out of global length of desired, target, input array
	if (k_x >= L_x) {
		return ;}
		
	float tempval { 0.f };
	float tempval_switch { 0.f };

	// k_x = 2^kj-1, j \in \lbrace \lfloor N/2^k \rfloor,... 2,1 \rbrace check
	if ( ((k_x%offset)==(offset-1)) && (k_x >= (offset-1)) ) {
		tempval_switch = f_in[k_x] ;
		tempval = f_in[k_x] + f_in[k_x-offset/2] ;
		f_out[k_x] = tempval ;
		f_out[k_x-offset/2] = tempval_switch; }
	else {
		tempval = f_in[k_x]; 
		f_out[k_x] = tempval ; }
}

__global__ void copy_swap(float* f_in, float* f_target, const int L_x) {
	const int k_x = threadIdx.x + blockIdx.x * blockDim.x ;
	
	// check if global thread index happens to fall out of global length of desired, target, input array
	if (k_x >= L_x) {
		return ; }
 
	float tempval = 0.f;
	tempval = f_in[k_x];
	f_in[k_x] = f_target[k_x];
	f_target[k_x] = tempval;
}

void Blelloch_scan_kernelLauncher(float* dev_f_in, float* dev_f_out, const int L_x, 
									const int M_in) {
	auto Nb = static_cast<int>(std::log2( L_x) );
	
	// sanity check
	std::cout << " In Blelloch_scan_kernelLauncher - this is Nb : " << Nb << std::endl;

	// determine number of thread blocks to be launched
	const int N_x { ( L_x + M_in - 1 ) / M_in } ;

	// do up sweep
	for (auto k = 1; k <= Nb; ++k) {
		Blelloch_up_global<<<N_x, M_in>>>( dev_f_in, dev_f_out, k, L_x) ; 
		copy_swap<<<N_x,M_in>>>(dev_f_in, dev_f_out,L_x); }
		
	// crucial step in Blelloch scan algorithm; copy the identity to the "end" of the array
	checkCudaErrors( 
		cudaMemset(&dev_f_in[(1<<Nb)-1], 0, sizeof(float)) );	
	

	// do down sweep
	for (auto k = Nb; k>=1; --k) {
		Blelloch_down_global<<<N_x,M_in>>>(dev_f_in, dev_f_out, k, L_x) ;
		copy_swap<<<N_x,M_in>>>( dev_f_in, dev_f_out, L_x) ; }

}		
				

	// Hillis-Steele (inclusive) scan 
__global__ void HillisSteele_global(float* f_in, float* f_out, const int k, const int L_x) {
	const int k_x = threadIdx.x + blockDim.x * blockIdx.x;
	const int offset { 1 << k };
	
	// check if global thread index happens to fall out of global length of desired, target, input array
	if (k_x >= L_x) {
		return ; }

	float tempval = 0.f;

	if (k_x >= offset) {
		tempval = f_in[k_x] + f_in[k_x-offset] ; }
	else {
		tempval = f_in[k_x] ; }
	f_out[k_x] = tempval;
}




void HillisSteele_kernelLauncher(float* dev_f_in, float* dev_f_out, const int L_x, 
									const int M_in) {
	auto Nb = static_cast<int>(std::log2( L_x) );
	
	// determine number of thread blocks to be launched
	const int N_x { ( L_x + M_in - 1 ) / M_in } ;
	
	for (auto k = 0; k < Nb; ++k) {
		HillisSteele_global<<<N_x, M_in>>>( dev_f_in, dev_f_out, k, L_x) ; 
		copy_swap<<<N_x,M_in>>>(dev_f_in, dev_f_out,L_x); }	
}		

// serial implementations

// blelloch_up - first part of Blelloch scan, the reduce or up sweep part
// k is the kth iteration, corresponding 1-to-1 to offset = 2^k
void blelloch_up( std::vector<float> f_in, std::vector<float> &f_out, const int k ) { 
	const int offset { 1 << k }; // offset = 2^k
	
	float tempval { 0.f };
	for (auto i = 0 ; i < f_in.size(); ++i) { 
		if ( ((i%offset)==(offset-1)) && (i >= (offset -1)) ) {
			tempval = f_in[i] + f_in[i-offset/2] ; }
		else {
			tempval = f_in[i]; }
		f_out[i] = tempval;
	}
}		


// blelloch_down - second part of Blelloch scan, the down sweep part 
void blelloch_down( std::vector<float> f_in, std::vector<float> &f_out, const int k) {
	
	const int offset { 1 << k }; // offset = 2^k
	
	float tempval { 0.f };
	float tempval_switch {0.f };
	
	for (auto i = 0; i < f_in.size(); ++i) {
		if ( ((i%offset)==(offset-1)) && (i >= (offset-1)) ) {
			tempval_switch = f_in[i] ;
			tempval = f_in[i] + f_in[i-offset/2] ;
			f_out[i] = tempval ;
			f_out[i-offset/2] = tempval_switch; }
		else {
			tempval = f_in[i]; 
			f_out[i] = tempval; }
	}
}	



//void blelloch_serial( std::vector<float> &f_in, std::vector<float> &f_out, const int N ) {
void blelloch_serial( std::vector<float> &f_in ) {
	std::vector<float> f_in_swap { f_in };
	std::vector<float> f_out { f_in };

	auto Nb = static_cast<int>(std::log2( f_in.size() )); 

	// up sweep, i.e. reduce part
	for (auto k = 1; k <= Nb; ++k) {
		blelloch_up(f_in_swap, f_out,k);
		f_in_swap = f_out ;}

	// Setting the "last entry's" value to the identity; for addition, addition is 0
	f_in_swap[(1<<Nb)-1] = 0;


	for (auto k = Nb; k >= 1; --k ) {
		blelloch_down(f_in_swap,f_out,k);
		f_in_swap = f_out ; }
	
	f_in = f_in_swap;
}

// Hillis-Steele scan which is an inclusive scan; loops through iterations to do the inclusive scan
void HillisSteele( std::vector<float> f_in, std::vector<float> &f_out, const int k) {
	const int offset { 1 << k }; // offset = 2^k
	
	float tempval { 0.f };
	for (auto i = 0; i < f_in.size(); ++i) {
		if (i >= offset) {
			tempval = f_in[i] + f_in[i-offset] ; }
		else {
			tempval = f_in[i] ; }
		f_out[i] = tempval;
	}
}

void HillisSteele_serial(std::vector<float> &f_in ) { 
	std::vector<float> f_in_swap { f_in } ;
	std::vector<float> f_out { f_in };

	auto Nb = static_cast<int>(std::log2( f_in.size() ));
	
	for (auto k = 0; k < Nb; ++k) {
		HillisSteele( f_in_swap, f_out, k ) ;
		f_in_swap = f_out;
	}
	f_in = f_in_swap;
}





