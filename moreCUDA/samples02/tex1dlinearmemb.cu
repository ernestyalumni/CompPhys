/*
 * tex1dlinearmemb.cu
 * 
 * cf. http://www.math.ntu.edu.tw/~wwang/mtxcomp2010/download/cuda_04_ykhung.pdf
 * from Yukai Hung a0934147@gmail.com Math Dept. National Taiwan Univ.
 * 
 * Compilation
 * 
 * nvcc -std=c++11 tex1dlinearmemb.cu -o tex1dlinearmemb
 * 
 * */
#include <iostream> // std::cout
#include <limits>  // std::numeric_limits
// Note below: the "copy" immediate below each line of "original" code is for playing around
// manually change values to test stuff out!

// declare texture reference (at file level)
texture<float,1,cudaReadModeElementType> texreference;
texture<float,cudaTextureType1D,cudaReadModeElementType> tex;

__global__ void kernel(float* doarray, int size)
{
	int index;
	
	// calculate each thread global index
	index=blockIdx.x*blockDim.x+threadIdx.x;
	
	// fetch global memory through texture reference
	doarray[index]=tex1Dfetch(texreference,index);
	
	return;
}
__global__ void kernel2(float* doarray, int size)
{
	// calculate each thread global index
	const int k_x = blockIdx.x*blockDim.x+threadIdx.x ;
	
	// fetch global memory through texture reference
	doarray[k_x]=tex1Dfetch(tex,k_x);
	
	return;
}



int main(int argc, char** argv)
{
	const int ARRAY_SIZE=3200;
	const int ARRAY_SIZE2=32; 

	constexpr const int M_in { 64 };
	
	float* harray;
	float* diarray;
	float* doarray;
	
	float xarray[ARRAY_SIZE2];
	float* dev_in;
	float* dev_out;
	
	// allocate host and device memory
	harray=(float*)malloc(sizeof(float)*ARRAY_SIZE);
	cudaMalloc((void**)&diarray,sizeof(float)*ARRAY_SIZE);
	cudaMalloc((void**)&doarray,sizeof(float)*ARRAY_SIZE);

	cudaMalloc((void**)&dev_in, sizeof(float)*ARRAY_SIZE2);
	cudaMalloc((void**)&dev_out, sizeof(float)*ARRAY_SIZE2);
	
	// initialize host array before usage
	for(int loop=0; loop<ARRAY_SIZE;loop++)
		harray[loop]=(float)rand()/(float) (RAND_MAX-1);

	for(int loop=0; loop<ARRAY_SIZE2;loop++) {
		xarray[loop]= (std::numeric_limits<float>::max()/(1000.f)) / pow(2,loop) ;
	}

	// sanity check: print out initial values:
	const int DISPLAY_SIZE = 22; // how may numbers you want to display, read out, or print out on screen
	static_assert( ARRAY_SIZE >= DISPLAY_SIZE, "ARRAY_SIZE needs to be equal or bigger than DISPLAY_SIZE");
	static_assert( ARRAY_SIZE2 >= DISPLAY_SIZE, "ARRAY_SIZE2 needs to be equal or bigger than DISPLAY_SIZE");

	std::cout << "Initially, " << std::endl;
	for (int i = 0; i < DISPLAY_SIZE; ++i) {
		std::cout << " " << harray[i] ; }
	std::cout << std::endl;

	std::cout << "\n and with xarray, initially, " << std::endl;
	for (int i = 0; i < DISPLAY_SIZE; ++i) {
		std::cout << " " << xarray[i] ; }
	std::cout << std::endl;


	std::cout << "numerical limit of float, minimum : " << std::numeric_limits<float>::min() << std::endl;
	std::cout << "numerical limit of float, maximum : " << std::numeric_limits<float>::max() << std::endl;
	std::cout << "numerical limit of float, lowest  : " << std::numeric_limits<float>::lowest() << std::endl;


		
	// copy array from host to device memory
	cudaMemcpy(diarray,harray,sizeof(float)*ARRAY_SIZE,cudaMemcpyHostToDevice);
	cudaMemcpy(dev_in,xarray,sizeof(float)*ARRAY_SIZE2,cudaMemcpyHostToDevice);

	// bind texture reference with linear memory
	cudaBindTexture(0,texreference,diarray,sizeof(float)*ARRAY_SIZE);
	cudaBindTexture(0,tex,dev_in,sizeof(float)*ARRAY_SIZE2);

	// execute device kernel
	kernel<<<(int)ceil((float)ARRAY_SIZE/64),64>>>(doarray,ARRAY_SIZE);
	kernel2<<<1,M_in>>>(dev_out,ARRAY_SIZE2);
	
	
	// unbind texture reference to free resource
	cudaUnbindTexture(texreference);
	cudaUnbindTexture(tex);
	
	// copy result array from device to host memory
	cudaMemcpy(harray,doarray,sizeof(float)*ARRAY_SIZE,cudaMemcpyDeviceToHost);
	cudaMemcpy(xarray,dev_out,sizeof(float)*ARRAY_SIZE2,cudaMemcpyDeviceToHost);

	// sanity check: print out, read out results
	std::cout << "After kernel, which has a tex1Dfetch, and cudaMemcpy, " << std::endl;
	for (int i = 0; i < DISPLAY_SIZE; ++i) {
		std::cout << " " << harray[i] ; }
	std::cout << std::endl;

	std::cout << "\n And for xarray : " << std::endl;
	for (int i = 0; i < DISPLAY_SIZE; ++i) {
		std::cout << " " << xarray[i] ; }
	std::cout << std::endl;

	
	// free host and device memory
	free(harray);
	cudaFree(diarray);
	cudaFree(doarray);

	cudaFree(dev_in);
	cudaFree(dev_out);
	
	return 0;
}
