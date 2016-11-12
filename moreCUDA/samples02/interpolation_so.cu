/*
 * interpolation_so.cu
 * 
 * cf. http://stackoverflow.com/questions/24789619/cuda-texture-interpolation-incorrect-for-normalized-co-ordinates
 * CUDA texture interpolation incorrect for normalized co-ordinates
 * from 
 * Sam http://stackoverflow.com/users/927046/sam
 * 
 * Compilation
 * 
 * nvcc -arch=sm_20 interpolation_so.cu // non-normalized coordinates
 * nvcc -arch=sm_20 interpolation_so.cu -DNORMALIZED // normalized coordinates, incorrect answer
 * 
 * */
 
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

texture<float, cudaTextureType1D, cudaReadModeElementType> table_tex;

const int N_table = 6;

// y = 2*x for x in [0,1)
float table[N_table] = {0, 0.4, 0.8, 1.2, 1.6, 2.0};

__global__ void hw_linear_interpolation(const float* inputs,
										float* interpolated,
										const unsigned int n_inputs)
{
	int tid = threadIdx.x;
	
	if (tid < n_inputs)
	{
		float val = inputs[tid];
		
#ifdef NORMALIZED
		float interp = tex1D(table_tex, val);
#else
		float interp = tex1D(table_tex, (N_table-1)*val+0.5f);
#endif
		interpolated[tid] = interp;
	}
}

int main(void)
{
	int N_inputs = 11;
	
	thrust::host_vector<float> h_inputs(N_inputs);
	thrust::device_vector<float> d_outputs(N_inputs);
	thrust::host_vector<float> h_outputs(N_inputs);
	
	// Allocate CUDA array in device memory to bind table_tex to.
	cudaChannelFormatDesc channelDesc = 
								cudaCreateChannelDesc<float>();
	
	cudaArray* cuArray_table;
	cudaMallocArray(&cuArray_table, &channelDesc, N_table, 0);
	
	// Copy to device memory some data located at address h_data
	// in host memory
	cudaMemcpyToArray(cuArray_table, 0, 0, table, N_table*sizeof(float),
						cudaMemcpyHostToDevice);
						
	// Initialize input values to interpolate from the table for.
	for (int i=0; i<N_inputs; i++) {
		h_inputs[i] = i*0.1f;
	}
	
	thrust::device_vector<float> d_inputs = h_inputs;
	
	// Set up texture for linear interpolation with normalized inputs.
	table_tex.addressMode[0] = cudaAddressModeClamp;
	table_tex.filterMode = cudaFilterModeLinear;
#ifdef NORMALIZED
	table_tex.normalized = true;
#else
	table_tex.normalized = false;
	std::cout << "\n We're in table_tex.normalized = false " << std::endl;
#endif

	cudaBindTextureToArray(table_tex, cuArray_table);
	hw_linear_interpolation<<<1,128>>>(
		thrust::raw_pointer_cast(d_inputs.data() ),
		thrust::raw_pointer_cast(d_outputs.data() ),
		N_inputs);
	cudaUnbindTexture(table_tex);
	h_outputs = d_outputs;
	
	std::cout << "     x     |   interp. y   |   actual y   ";
	std::cout << std::endl;
	std::cout << "------------------------------------------";
	std::cout << std::endl;
	
	std::cout.setf(std::ios::fixed, std::ios::floatfield);
	for (int i=0; i<N_inputs; i++)
	{
		std::cout << "    ";
		std::cout.precision(1);
		std::cout.width(3);
		std::cout << h_inputs[i];
		std::cout << "    |";
		
		std::cout << "     ";
		std::cout.precision(5);
		std::cout.width(7);
		std::cout << h_outputs[i];
		std::cout << "    |";
		
		std::cout << "   ";
		std::cout.width(7);
		std::cout << 2*(i*0.1f);
		std::cout << std::endl;
	}
	return 0;
}
	
	
