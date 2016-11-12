/*
 * tex1dlinearmem.cu
 * 
 * cf. http://www.math.ntu.edu.tw/~wwang/mtxcomp2010/download/cuda_04_ykhung.pdf
 * from Yukai Hung a0934147@gmail.com Math Dept. National Taiwan Univ.
 * 
 * Compilation
 * 
 * nvcc -arch=sm_20 interpolation_so.cu // non-normalized coordinates
 * nvcc -arch=sm_20 interpolation_so.cu -DNORMALIZED // normalized coordinates, incorrect answer
 * 
 * */

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
	int size=3200;
	
	float* harray;
	float* diarray;
	float* doarray;
	
	// allocate host and device memory
	harray=(float*)malloc(sizeof(float)*size);
	cudaMalloc((void**)&diarray,sizeof(float)*size);
	cudaMalloc((void**)&doarray,sizeof(float)*size);
	
	// initialize host array before usage
	for(int loop=0; loop<size;loop++)
		harray[loop]=(float)rand()/(float) (RAND_MAX-1);
		
	// copy array from host to device memory
	cudaMemcpy(diarray,harray,sizeof(float)*size,cudaMemcpyHostToDevice);
	
	// bind texture reference with linear memory
	cudaBindTexture(0,texreference,diarray,sizeof(float)*size);
	
	// execute device kernel
	kernel<<<(int)ceil((float)size/64),64>>>(doarray,size);
	
	// unbind texture reference to free resource
	cudaUnbindTexture(texreference);
	
	// copy result array from device to host memory
	cudaMemcpy(harray,doarray,sizeof(float)*size,cudaMemcpyDeviceToHost);
	
	// free host and device memory
	free(harray);
	cudaFree(diarray);
	cudaFree(doarray);
	
	return 0;
}
