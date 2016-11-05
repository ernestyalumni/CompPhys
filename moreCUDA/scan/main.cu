/* main.cu
 * Ernest Yeung
 * ernestyalumni@gmail.com
 * Demonstrates Hillis/Steele and Blelloch (exclusive) scan with a parallel implementation
 * with CUDA C/C++ and global memory
 * 
 * */
#include <vector> // std::vector
#include <cmath> // std::log2
#include <chrono> // chrono::steady_clock::now() 

#include "./common/timer.h"  // GpuTimer

#include "./methods/checkerror.h"
#include "./methods/scans.h" /* Blelloch_up_global, Blelloch_down_global, copy_swap
								* Blelloch_scan_kernelLauncher, 
								* HillisSteele_global, HillisSteele_kernelLauncher */

int main() {
	// "boilerplate"
	// initiate correct GPU
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	if (deviceCount == 0) {
		exit(EXIT_FAILURE);
	}
	int dev = 0;
	cudaSetDevice(dev);
	
	cudaDeviceProp devProps;
	if (cudaGetDeviceProperties(&devProps, dev) == 0) {
		std::cout << " Using device " << dev << ":\n" ;
		std::cout << devProps.name << "; global mem: " << (int)devProps.totalGlobalMem <<
			"; compute v" << (int)devProps.major << "." << (int)devProps.minor << "; clock: " <<
			(int)devProps.clockRate << " kHz" << std::endl; }
	// END if GPU properties
	
	// MANUALLY CHANGE THESE 3
	// input array with interesting values "boilerplate"
	const int ARRAY_SIZE { 1<<20 } ;  /* 2^20=1048576 worked on GTX 980 Ti; for larger, such as 2^22, Segmentation Fault
										* for only the Blelloch scan.  
										* For Hillis/Steele, 2^20=1048576 worked on GTX 980 Ti; for larger, such as 2^21, 
										* CUDA error
										* Note that it was critical that M_x is maxed out (on GTX 980 Ti, 
										* it's 1024 max. number of threads per block
										* */
//	const long ARRAY_SIZE { 1<< 29 }; // this this line if needed
	const int ARRAY_BYTES { ARRAY_SIZE * sizeof(float) } ;
//	const long ARRAY_BYTES { ARRAY_SIZE * sizeof(float) }; // use this line if needed
	const int L_x { ARRAY_SIZE } ;
	std::cout << "For an (float) array of size (length) : " << ARRAY_SIZE << std::endl ;
	std::cout << "or, in bytes, " << ARRAY_BYTES << std::endl;
	
	
	const int DISPLAY_SIZE = 22; // how many numbers you want to display, read out, or print out on screen
	static_assert( ARRAY_SIZE >= DISPLAY_SIZE, "ARRAY_SIZE needs to be equal or bigger than DISPLAY_SIZE");
	
	// generate input array on host
	std::vector<float> f_vec;
	for (int i = 0; i < ARRAY_SIZE; ++i) {
		f_vec.push_back(i+1) ; }
	float* host_f_in;
	host_f_in = f_vec.data();
	// END of initializing, creating input array with interesting values, on host CPU, boilerplate

	// sanity check print out of initial values: 
	std::cout << " Initially, " << std::endl;
	for (int i = 0 ; i < DISPLAY_SIZE; ++i)  {
		std::cout << " " << f_vec[i] ; }
	std::cout << std::endl;
	
	// declare GPU memory pointers
	float *dev_f_in, *dev_f_out;
	
	// allocate GPU memory
	checkCudaErrors(
		cudaMalloc((void **) &dev_f_in, ARRAY_BYTES ));
	checkCudaErrors(
		cudaMalloc((void **) &dev_f_out, ARRAY_BYTES));

	// transfer the input array to the GPU
	cudaMemcpy(dev_f_in, host_f_in, ARRAY_BYTES, cudaMemcpyHostToDevice);
	
	// MANUALLY CHANGE THIS 1
	// launch the kernel
	// input parameters for number of threads per block, M_x, and number of blocks, N_x
	const int M_x {1024};
	
	// Blelloch scan (exclusive scan) in parallel
	// time the kernel
	GpuTimer timer;
	timer.Start();

	Blelloch_scan_kernelLauncher(dev_f_in, dev_f_out, L_x, M_x) ;

	timer.Stop();
	
	cudaDeviceSynchronize(); 
	checkCudaErrors(
		cudaGetLastError() );
		
	std::cout<< "Blelloch scan, in parallel, ran in : " << timer.Elapsed() << " msecs. " << std::endl;
	
	std::cout << "After Blelloch scan : " << std::endl;
		
	// copy our results from device to host
	float host_f_out[ARRAY_SIZE];
	checkCudaErrors( cudaMemcpy(host_f_out, dev_f_in, ARRAY_BYTES, cudaMemcpyDeviceToHost) );
	// read out results into our useful vector
	f_vec.insert(f_vec.begin(), &host_f_out[0], &host_f_out[ARRAY_SIZE] );
	// print out results
	for (int i = 0 ; i < DISPLAY_SIZE; ++i)  {
		std::cout << " " << f_vec[i] ; }
	std::cout << std::endl;
	// uncomment for printout of previous step
/*	checkCudaErrors( 
		cudaMemcpy(host_f_out, dev_f_out, ARRAY_BYTES, cudaMemcpyDeviceToHost) );
	f_vec.insert(f_vec.begin(), &host_f_out[0], &host_f_out[ARRAY_SIZE] );
	for (int i = 0 ; i < DISPLAY_SIZE; ++i)  {
		std::cout << " " << f_vec[i] ; }  */
	std::cout << std::endl;

	// Hillis/Steele scan, in parallel
	// transfer the input array to the GPU
	for (int i = 0; i < ARRAY_SIZE; ++i) {
		f_vec[i] = i  ; }
	host_f_in = f_vec.data();

	cudaMemcpy(dev_f_in, host_f_in, ARRAY_BYTES, cudaMemcpyHostToDevice);

	// time the Hillis/Steele scan, in parallel, kernel
	timer.Start();

	HillisSteele_kernelLauncher(dev_f_in, dev_f_out, L_x, M_x) ;

	timer.Stop();
	
	cudaDeviceSynchronize(); 
	checkCudaErrors(
		cudaGetLastError() );
	
	std::cout<< "Hillis/Steele scan, in parallel, ran in : " << timer.Elapsed() << " msecs. " << std::endl;
	
	std::cout << "After Hillis/Steele scan : " << std::endl;

		// copy our results from device to host
	checkCudaErrors( cudaMemcpy(host_f_out, dev_f_in, ARRAY_BYTES, cudaMemcpyDeviceToHost) );
	// read out results into our useful vector
	f_vec.insert(f_vec.begin(), &host_f_out[0], &host_f_out[ARRAY_SIZE] );
	// print out results
	for (int i = 0 ; i < DISPLAY_SIZE; ++i)  {
		std::cout << " " << f_vec[i] ; }
	std::cout << std::endl;
	// uncomment for printout of previous step
/*	checkCudaErrors( 
		cudaMemcpy(host_f_out, dev_f_out, ARRAY_BYTES, cudaMemcpyDeviceToHost) );
	f_vec.insert(f_vec.begin(), &host_f_out[0], &host_f_out[ARRAY_SIZE] );
	for (int i = 0 ; i < DISPLAY_SIZE; ++i)  {
		std::cout << " " << f_vec[i] ; } */
	std::cout << std::endl;

	// Blelloch scan, in serial
	// vector for serial implementation
//	std::vector<float> f_vec_out( ARRAY_SIZE,0); // I obtain error : name followed by "::" must be a class or namespace name
//          detected during:
 //           instantiation of class "std::__iterator_traits<_Iterator, void> [with _Iterator=int]" 

//	std::vector<float> f_vec_out;

	for (int i = 0; i < ARRAY_SIZE; ++i) {
		f_vec[i] = ((float) (i+1))  ; }	
	// sanity check print out 
	std::cout << " For the serial implementation of Blelloch scan, initially : " << std::endl;
	for (int i = 0; i <DISPLAY_SIZE;++i) {
		std::cout << f_vec[i] << " " ; }
	std::cout << std::endl;

	// For measuring execution time of a piece of code, use now() function of chrono's steady_clock
	auto start = std::chrono::steady_clock::now(); 

	blelloch_serial( f_vec);

	auto end = std::chrono::steady_clock::now(); 

	// print out results
	for (int i = 0 ; i < DISPLAY_SIZE; ++i)  {
		std::cout << f_vec[i] << " " ; } 
	std::cout << std::endl;

/*	for (int i = 0 ; i < DISPLAY_SIZE; ++i)  {
		std::cout << " " << f_vec_out[i] ; } 
	std::cout << std::endl;
*/
	auto diff = end - start;
	std::cout << " Blelloch scan, in serial, ran in : " << 
		std::chrono::duration <double, std::milli>(diff).count() << " ms " << std::endl ; 

	// Hillis-Steele (inclusion) scan, serial
	for (int i = 0; i < ARRAY_SIZE; ++i) {
		f_vec[i] = ((float) i )  ; }	
	// sanity check print out 
	std::cout << " For the serial implementation of Hillis-Steele scan, initially : " << std::endl;
	for (int i = 0; i <DISPLAY_SIZE;++i) {
		std::cout << f_vec[i] << " " ; }
	std::cout << std::endl;


	start = std::chrono::steady_clock::now(); 
	HillisSteele_serial( f_vec );
	end = std::chrono::steady_clock::now(); 

	// print out results
	for (int i = 0 ; i < DISPLAY_SIZE; ++i)  {
		std::cout << f_vec[i] << " " ; } 
	std::cout << std::endl;
	
	diff = end - start;
	std::cout << " Hillis-Steele scan, in serial, ran in : " << 
		std::chrono::duration <double, std::milli>(diff).count() << " ms " << std::endl ; 
	

	// free GPU memory
	cudaFree( dev_f_in  );
	cudaFree( dev_f_out );

}
