/*
 * tex1dlinearmemc.cu
 * 
 * cf. http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#abstract  
 * 3.2.11.1.1. Texture Object API of CUDA Toolkit v8.0 Documentation, Programming Guide
 * 
 * Compilation
 * 
 * nvcc -std=c++11 simpletransform.cu -o simpletransform
 * 
 * */
#include <iostream> // std::cout
// Simple transformation kernel
__global__ void transformKernel(float* output, 
								cudaTextureObject_t texObj,
								int width, int height,
								float theta)
{
	// Calculate normlized texture coordinates
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	
	float u = x / (float) width;
	float v = y / (float) height;
	
	// Transform coordinates
	u -= 0.5f;
	v -= 0.5f;
	float tu = u * cosf(theta) - v * sinf(theta) + 0.5f;
	float tv = v * cosf(theta) + u * sinf(theta) + 0.5f;
	
	// Read from texture and write to global memory
	output[y * width + x] = tex2D<float>(texObj, tu, tv);
}

// Host code
int main()
{
	// Allocate CUDA array in device memory
	cudaChannelFormatDesc channelDesc = 
		cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	
	cudaArray* cuArray;
	cudaMallocArray(&cuArray, &channelDesc, width, height);
	
	// Copy to device memory some data located at address h_data 
	// in host memory
	cudaMemcpyToArray(cuArray, 0, 0, h_data, size,
						cudaMemcpyHostToDevice);
	
	// Specify texture
	struct cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc) );
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = cuArray;
	
	// Specify texture object parameters
	struct cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0]  = cudaAddressModeWrap;
	texDesc.addressMode[1]  = cudaAddressModeWrap;
	texDesc.filterMode       = cudaFilterModeLinear;
	texDesc.readMode         = cudaReadModeElementType;
	texDesc.normalizedCoords = 1;
	
	// Create texture object
	cudaTextureObject_t texObj  = 0;
	cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
	
	// Allocate result of transformation in device memory
	float* output;
	cudaMalloc(&output, width * height * sizeof(float));
	
	// Invoke kernel
	dim3 dimBlock(16,16);
	dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, 
				(height + dimBlock.y - 1) / dimBlock.y);
	transformKernel<<<dimGrid, dimBlock>>>(output, 
											texObj, width, height,
											angle);
											
	// Destroy texture object
	cudaDestroyTextureObject(texObj);
	
	// Free device memory
	cudaFreeArray(cuArray);
	cudaFree(output);
	
	return 0;
}
							

