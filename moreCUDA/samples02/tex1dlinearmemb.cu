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

// declare texture reference (at file level)
texture<float,1,cudaReadModeElementType> texreference;

__global__ void kernel(float* doarray, int size)
{
	int index;
	
	// calculate each thread global index
	index=blockIdx.x*blockDim.x+threadIdx.x;
	
	// fetch global memory through texture reference
	doarray[index]=tex1Dfetch(texreference,index);
	
	return;
}

int main(int argc, char** argv)
{
	const int ARRAY_SIZE=3200;
	
	float* harray;
	float* diarray;
	float* doarray;
	
	// allocate host and device memory
	harray=(float*)malloc(sizeof(float)*ARRAY_SIZE);
	cudaMalloc((void**)&diarray,sizeof(float)*ARRAY_SIZE);
	cudaMalloc((void**)&doarray,sizeof(float)*ARRAY_SIZE);
	
	// initialize host array before usage
	for(int loop=0; loop<ARRAY_SIZE;loop++)
		harray[loop]=(float)rand()/(float) (RAND_MAX-1);


	// sanity check: print out initial values:
	const int DISPLAY_SIZE = 22; // how may numbers you want to display, read out, or print out on screen
	static_assert( ARRAY_SIZE >= DISPLAY_SIZE, "ARRAY_SIZE needs to be equal or bigger than DISPLAY_SIZE");
	
	std::cout << "Initially, " << std::endl;
	for (int i = 0; i < DISPLAY_SIZE; ++i) {
		std::cout << " " << harray[i] ; }
	std::cout << std::endl;


		
	// copy array from host to device memory
	cudaMemcpy(diarray,harray,sizeof(float)*ARRAY_SIZE,cudaMemcpyHostToDevice);
	
	// bind texture reference with linear memory
	cudaBindTexture(0,texreference,diarray,sizeof(float)*ARRAY_SIZE);
	
	// execute device kernel
	kernel<<<(int)ceil((float)ARRAY_SIZE/64),64>>>(doarray,ARRAY_SIZE);
	
	// unbind texture reference to free resource
	cudaUnbindTexture(texreference);
	
	// copy result array from device to host memory
	cudaMemcpy(harray,doarray,sizeof(float)*ARRAY_SIZE,cudaMemcpyDeviceToHost);


	// sanity check: print out, read out results
	std::cout << "After kernel, which has a tex1Dfetch, and cudaMemcpy, " << std::endl;
	for (int i = 0; i < DISPLAY_SIZE; ++i) {
		std::cout << " " << harray[i] ; }
	std::cout << std::endl;

	
	// free host and device memory
	free(harray);
	cudaFree(diarray);
	cudaFree(doarray);
	
	return 0;
}
