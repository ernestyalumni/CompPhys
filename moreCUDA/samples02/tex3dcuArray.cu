/*
 * tex3dcuArray.cu
 * 3-dimension cuda array example
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
texture<float,3,cudaReadModeElementType> texreference;

__global__ void kernel(float* dmatrix, int size)
{
	int loop;
	int xindex;
	int yindex;
	int zindex;
	
	// calculate each thread global index
	xindex=blockIdx.x*blockDim.x+threadIdx.x;
	yindex=blockIdx.y*blockDim.y+threadIdx.y;
	
	for (loop=0;loop<size;loop++)
	{
		zindex=loop;
	
		// fetch cuda array through texture reference
		dmatrix[zindex*size*size + yindex*size+xindex]=
			tex3D(texreference,xindex,yindex,zindex);
	}
	return;
}

int main(int argc, char** argv)
{
	int size=256;

	dim3 blocknum;
	dim3 blocksize;
	
	float* hmatrix;
	float* dmatrix;
	
	cudaArray* cudaarray;
	cudaExtent volumesize;
	cudaChannelFormatDesc channel;
	
	cudaMemcpy3DParms copyparms={0};
	
	// allocate host and device memory
	hmatrix=(float*)malloc(sizeof(float)*size*size*size);
	cudaMalloc((void**)&dmatrix,sizeof(float)*size*size*size);
	
	// initialize host array before usage
	for(int loop=0; loop<size*size*size;loop++)
		hmatrix[loop]=(float)rand()/(float) (RAND_MAX-1);

	// set cuda array volume size
	volumesize=make_cudaExtent(size,size,size);	
		
	// create channel to describe data type
	channel=cudaCreateChannelDesc<float>();	

	// allocate device memory for cuda array
	cudaMalloc3DArray(&cudaarray,&channel,volumesize);

	// set cuda array copy parameters
	copyparms.extent=volumesize;
	copyparms.dstArray=cudaarray;
	copyparms.kind=cudaMemcpyHostToDevice;
	
	copyparms.srcPtr= make_cudaPitchedPtr((void*)hmatrix,sizeof(float)*size,size,size);
	
	cudaMemcpy3D(&copyparms);

	// set texture filter mode property
	// use cudaFilterModePoint or cudaFilterModeLinear
	texreference.filterMode=cudaFilterModePoint;

	// set texture address mode property
	// use cudaAddressModeClamp or cudaAddressModeWrap
	texreference.addressMode[0]=cudaAddressModeWrap;
	texreference.addressMode[1]=cudaAddressModeWrap;
	texreference.addressMode[2]=cudaAddressModeClamp;

	// bind texture reference with cuda array
	cudaBindTextureToArray(texreference,cudaarray,channel);

	
	blocksize.x=8;
	blocksize.y=8;
	blocksize.z=8;
	
	blocknum.x=(int)ceil((float)size/8);
	blocknum.y=(int)ceil((float)size/8);
	blocknum.z=(int)ceil((float)size/8);
	
		
	// execute device kernel
	kernel<<<blocknum,blocksize>>>(dmatrix,size);
	
	// unbind texture reference to free resource
	cudaUnbindTexture(texreference);
	
	// copy result array from device to host memory
	const int bytes = sizeof(float)*size*size*size;
	cudaMemcpy(hmatrix,dmatrix,bytes,cudaMemcpyDeviceToHost);
	
	// free host and device memory
	free(hmatrix);
	cudaFree(dmatrix);
	cudaFreeArray(cudaarray);
	
	return 0;
}

