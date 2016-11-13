/*
 * tex2dcuArray.cu
 * 2-dimension cuda array example
 * 
 * cf. http://www.math.ntu.edu.tw/~wwang/mtxcomp2010/download/cuda_04_ykhung.pdf
 * from Yukai Hung a0934147@gmail.com Math Dept. National Taiwan Univ.
 * 
 * Compilation
 * 
 * nvcc -std=c++11 tex1dlinearmemb.cu -o tex1dlinearmemb
 * 
 * */
// declare texture reference (at file level)
texture<float,2,cudaReadModeElementType> texreference;

__global__ void kernel(float* dmatrix, int size)
{
	int xindex;
	int yindex;
	
	// calculate each thread global index
	xindex=blockIdx.x*blockDim.x+threadIdx.x;
	yindex=blockIdx.y*blockDim.y+threadIdx.y;
	
	// fetch cuda array through texture reference
	dmatrix[yindex*size+xindex]=tex2D(texreference,xindex,yindex);
	
	return;
}

int main(int argc, char** argv)
{
	int size=3200;

	dim3 blocknum;
	dim3 blocksize;
	
	float* hmatrix;
	float* dmatrix;
	
	cudaArray* carray;
	cudaChannelFormatDesc channel;
	
	// allocate host and device memory
	hmatrix=(float*)malloc(sizeof(float)*size*size);
	cudaMalloc((void**)&dmatrix,sizeof(float)*size*size);
	
	// initialize host array before usage
	for(int loop=0; loop<size*size;loop++)
		hmatrix[loop]=(float)rand()/(float) (RAND_MAX-1);
		
	// create channel to describe data type
	channel=cudaCreateChannelDesc<float>();	

	// allocate device memory for cuda array
	cudaMallocArray(&carray,&channel,size,size);
		
	// copy matrix from host to device memory
	const int bytes=sizeof(float)*size*size;
	cudaMemcpyToArray(carray,0,0,hmatrix,bytes,cudaMemcpyHostToDevice);
	
	// set texture filter mode property
	// use cudaFilterModePoint or cudaFilterModeLinear
	texreference.filterMode=cudaFilterModePoint;
	
	// set texture address mode property
	// use cudaAddressModeClamp or cudaAddressModeWrap
	texreference.addressMode[0]=cudaAddressModeWrap;
	texreference.addressMode[1]=cudaAddressModeClamp;
	
	// bind texture reference with cuda array
	cudaBindTextureToArray(texreference,carray);
	
	blocksize.x=16;
	blocksize.y=16;

	blocknum.x=(int)ceil((float)size/16);
	blocknum.y=(int)ceil((float)size/16);
	
	// execute device kernel
	kernel<<<blocknum,blocksize>>>(dmatrix,size);
	
	// unbind texture reference to free resource
	cudaUnbindTexture(texreference);
	
	// copy result array from device to host memory
	cudaMemcpy(hmatrix,dmatrix,bytes,cudaMemcpyDeviceToHost);
	
	// free host and device memory
	free(hmatrix);
	cudaFree(dmatrix);
	cudaFreeArray(carray);
	
	return 0;
}

