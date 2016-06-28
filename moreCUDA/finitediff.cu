/* finitediff.cu
 * finite-difference
 * based on Mark Harris' finite-difference.cu script
 * Ernest Yeung  ernestyalumni@gmail.com
 * 20160627
*/
#include <stdio.h>
#include <assert.h>

#include "./common/errors.h" // Convenience function for checking CUDA runtime API results
							// can be wrapped around any runtime API call.  No-op in release builds.
float fx = 1.0f, fy = 1.0f, fz = 1.0f;
const int m_x = 64, m_y = 64, m_z = 64;

// shared memory tiles will be M*-by-*Pencils
// sPencils is used when each thread calculates the derivative at one point
// lPencils is used for coalescing in y and z where each thread has to
// calculate the derivative at multiple points
const int sPencils = 4;  // small # pencils
const int lPencils = 32; // large # pencils 

dim3 grid[3][2], block[3][2];

// stencil coefficients  // __constant__ uses constant memory on device (GPU), FTW!
__constant__ float c_ax, c_bx, c_cx, c_dx;
__constant__ float c_ay, c_by, c_cy, c_dy;
__constant__ float c_az, c_bz, c_cz, c_dz;

// host routine to set constant data
void setDerivativeParameters()
{
	// check to make sure dimensions are integral multiples of sPencils
	if ((m_x % sPencils != 0) || (m_y % sPencils != 0) || (m_z % sPencils != 0)) {
		printf("'m_x', 'm_y', and 'm_z' must be integral multiples of sPencils\n");
		exit(1);
	}
	
	if ((m_x % lPencils != 0) || (m_y % lPencils != 0)) {
		printf("'m_x' and 'm_y' must be multiples of lPencils\n");
		exit(1);
	}
	
	// stencil weights (for unit length problem)
	float dsinv = m_x - 1.f;
	
	float ax =  4.f / 5.f   * dsinv;
	float bx = -1.f / 5.f   * dsinv;
	float cx =  4.f / 105.f * dsinv;
	float dx = -1.f / 280.f * dsinv;
	HANDLE_ERROR(
		cudaMemcpyToSymbol(c_ax, &ax, sizeof(float), 0, cudaMemcpyHostToDevice) );
	HANDLE_ERROR(
		cudaMemcpyToSymbol(c_bx, &bx, sizeof(float), 0, cudaMemcpyHostToDevice) );
	HANDLE_ERROR(
		cudaMemcpyToSymbol(c_cx, &cx, sizeof(float), 0, cudaMemcpyHostToDevice) );
	HANDLE_ERROR(
		cudaMemcpyToSymbol(c_dx, &dx, sizeof(float), 0, cudaMemcpyHostToDevice) );
		
	dsinv = m_y - 1.f;

	float ay =  4.f / 5.f   * dsinv;
	float by = -1.f / 5.f   * dsinv;
	float cy =  4.f / 105.f * dsinv;
	float dy = -1.f / 280.f * dsinv;
	HANDLE_ERROR(
		cudaMemcpyToSymbol(c_ay, &ay, sizeof(float), 0, cudaMemcpyHostToDevice) );
	HANDLE_ERROR(
		cudaMemcpyToSymbol(c_by, &by, sizeof(float), 0, cudaMemcpyHostToDevice) );
	HANDLE_ERROR(
		cudaMemcpyToSymbol(c_cy, &cy, sizeof(float), 0, cudaMemcpyHostToDevice) );
	HANDLE_ERROR(
		cudaMemcpyToSymbol(c_dy, &dy, sizeof(float), 0, cudaMemcpyHostToDevice) );

	dsinv = m_z - 1.f;

	float az =  4.f / 5.f   * dsinv;
	float bz = -1.f / 5.f   * dsinv;
	float cz =  4.f / 105.f * dsinv;
	float dz = -1.f / 280.f * dsinv;
	HANDLE_ERROR(
		cudaMemcpyToSymbol(c_az, &az, sizeof(float), 0, cudaMemcpyHostToDevice) );
	HANDLE_ERROR(
		cudaMemcpyToSymbol(c_bz, &bz, sizeof(float), 0, cudaMemcpyHostToDevice) );
	HANDLE_ERROR(
		cudaMemcpyToSymbol(c_cz, &cz, sizeof(float), 0, cudaMemcpyHostToDevice) );
	HANDLE_ERROR(
		cudaMemcpyToSymbol(c_dz, &dz, sizeof(float), 0, cudaMemcpyHostToDevice) );

	// Execution configurations for small and large pencil tiles
	
	grid[0][0]  = dim3(m_y / sPencils, m_z, 1);
	block[0][0] = dim3(m_x , sPencils, 1);
	
	grid[0][1]  = dim3(m_y / lPencils, m_z, 1);
	block[0][1] = dim3(m_x , sPencils, 1);
	
	grid[1][0]  = dim3(m_x / sPencils, m_z, 1);
	block[1][0] = dim3(sPencils , m_y, 1);
	
	grid[1][1]  = dim3(m_x / lPencils, m_z, 1);
	// we want to use the same number of threads as above,
	// so when we use lPencils instead of sPencils in one
	// dimension, we multiply the other by sPencils/lPencils
	block[1][1] = dim3(lPencils, m_y * sPencils / lPencils, 1);
	
	grid[2][0]  = dim3(m_x / sPencils, m_y, 1);
	block[2][0] = dim3(sPencils, m_z, 1);
	
	grid[2][1]  = dim3(m_x / lPencils, m_y, 1);
	block[2][1] = dim3(lPencils, m_z * sPencils / lPencils, 1);
}

void initInput(float *f, int dim)
{
	const float twopi = 8.f * (float)atan(1.0);
	
	for (int k = 0; k < m_z; k++) {
		for (int j = 0; j < m_y; j++) {
			for (int i = 0; i < m_x; i++) {
				switch(dim) {
					case 0:
						f[k*m_x*m_y+j*m_x+i] = cos(fx*twopi*(i-1.f)/(m_x-1.f));
						break;
					case 1:
						f[k*m_x*m_y+j*m_x+i] = cos(fy*twopi*(j-1.f)/(m_y-1.f));
						break;
					case 2:
						f[k*m_x*m_y+j*m_x+i] = cos(fz*twopi*(k-1.f)/(m_z-1.f));
						break;
				}
			}
		}
	}
}

void initSol(float *sol, int dim)
{
	const float twopi = 8.f * (float)atan(1.0);
	
	for (int k = 0; k < m_z; k++) {
		for (int j = 0; j < m_y; j++) {
			for (int i = 0; i < m_x; i++) {
				switch(dim) {
					case 0:
						sol[k*m_x*m_y+j*m_x+i] = -fx*twopi*sin(fx*twopi*(i-1.f)/(m_x-1.f));
						break;
					case 1:
						sol[k*m_x*m_y+j*m_x+i] = -fy*twopi*sin(fy*twopi*(j-1.f)/(m_y-1.f));
						break;
					case 2:
						sol[k*m_x*m_y+j*m_x+i] = -fz*twopi*sin(fz*twopi*(k-1.f)/(m_z-1.f));
						break;
				}
			}
		}
	}
}

void checkResults(double &error, double &maxError, float *sol, float *df)
{
	// error = sqrt(sum((sol-df)**2)/(M_x*M_y*M_z))
	// maxError = maxval(abs(sol-df))
	maxError = 0;
	error = 0;
	for (int k = 0; k < m_z; k++) {
		for (int j = 0; j < m_y; j++) {
			for (int i = 0; i < m_x; i++) {
				float s = sol[k*m_x*m_y+j*m_x+i];
				float f = df[k*m_x*m_y+j*m_x+i];
				// printf("%d %d %d: %f %f\n", i, j, k, s, f);
				error += (s-f)*(s-f);
				if (fabs(s-f) > maxError) maxError = fabs(s-f);
			}
		}
	}
	error = sqrt(error / (m_x*m_y*m_z));
}

// ---------------
// x derivatives
// ---------------

__global__ void derivative_x(float *f, float *df)
{
	__shared__ float s_f[sPencils][m_x+8]; // 4-wide halo
	
	int i  = threadIdx.x;
	int j  = blockIdx.x*blockDim.y + threadIdx.y;
	int k  = blockIdx.y;
	int si = i + 4;       // local i for shared memory access + halo offset
	int sj = threadIdx.y; // local j for shared memory access
						
	int globalIdx = k * m_x * m_y + j * m_x + i;
	
	s_f[sj][si] = f[globalIdx];
	
	__syncthreads();
	
	// fill in periodic images in shared memory array
	if (i < 4) {
		s_f[sj][si-4]  = s_f[sj][si+m_x-5];
		s_f[sj][si+m_x] = s_f[sj][si+1];
	}
	
	__syncthreads();
	
	df[globalIdx] = 
		( c_ax * ( s_f[sj][si+1] - s_f[sj][si-1] )
		* c_bx * ( s_f[sj][si+2] - s_f[sj][si-2] )
		* c_cx * ( s_f[sj][si+3] - s_f[sj][si-3] )
		* c_dx * ( s_f[sj][si+4] - s_f[sj][si-4] ) );
}


// this version uses a 64x32 shared memory tile.
// still with 64*sPencils threads

__global__ void derivative_x_lPencils(float *f, float *df)
{
	__shared__ float s_f[lPencils][m_x+8]; // 4-wide halo
	
	int i     = threadIdx.x;
	int jBase = blockIdx.x*lPencils;
	int k     = blockIdx.y;
	int si    = i + 4; // local i for shared memory access + halo offset
	
	for (int sj = threadIdx.y; sj < lPencils; sj += blockDim.y) {
		int globalIdx = k * m_x * m_y + (jBase + sj) * m_x + i;
		s_f[sj][si] = f[globalIdx];
	}
	
	__syncthreads();
	
	// fill in periodic images in shared memory array
	if (i < 4) {
		for (int sj = threadIdx.y; sj < lPencils; sj += blockDim.y) {
			s_f[sj][si-4]   = s_f[sj][si+m_x-5];
			s_f[sj][si+m_x] = s_f[sj][si+1];
		}
	}
	
	__syncthreads();
	
	for (int sj = threadIdx.y; sj < lPencils; sj += blockDim.y) {
		int globalIdx = k * m_x * m_y + (jBase + sj) * m_x + i ;
		df[globalIdx] = 
			( c_ax * ( s_f[sj][si+1] - s_f[sj][si-1] )
			+ c_bx * ( s_f[sj][si+2] - s_f[sj][si-2] )
			+ c_cx * ( s_f[sj][si+3] - s_f[sj][si-3] )
			+ c_dx * ( s_f[sj][si+4] - s_f[sj][si-4] ) );
	}
}

// ------------- 
// y derivatives
// -------------

__global__ void derivative_y(float *f, float *df)
{
	__shared__ float s_f[m_y+8][sPencils];
	
	int i  = blockIdx.x*blockDim.x + threadIdx.x;
	int j  = threadIdx.y;
	int k  = blockIdx.y;
	int si = threadIdx.x;
	int sj = j + 4;
	
	int globalIdx = k * m_x * m_y + j * m_x + i;
	
	s_f[sj][si] = f[globalIdx];
	
	__syncthreads();
	
	if (j < 4) {
		s_f[sj-4][si]   = s_f[sj+m_y-5][si];
		s_f[sj+m_y][si] = s_f[sj+1][si];
	}
	
	__syncthreads();
	
	df[globalIdx] = 
		( c_ay * ( s_f[sj+1][si] - s_f[sj-1][si] )
		+ c_by * ( s_f[sj+2][si] - s_f[sj-2][si] )
		+ c_cy * ( s_f[sj+3][si] - s_f[sj-3][si] )
		+ c_dy * ( s_f[sj+4][si] - s_f[sj-4][si] ) );
}

// y derivative using a tile of 32x64,
// launch with thread block of 32x8
__global__ void derivative_y_lPencils(float *f, float *df)
{
	__shared__ float s_f[m_y+8][lPencils];
	
	int i  = blockIdx.x*blockDim.x + threadIdx.x;
	int k  = blockIdx.y;
	int si = threadIdx.x;
	
	for (int j = threadIdx.y; j < m_y; j += blockDim.y) {
		int globalIdx = k * m_x * m_y + j * m_x + i;
		int sj = j + 4;
		s_f[sj][si] = f[globalIdx];
	}
	
	__syncthreads();
	
	int sj = threadIdx.y + 4;
	if (sj < 8) { 
		s_f[sj-4][si]   = s_f[sj+m_y-5][si];
		s_f[sj+m_y][si] = s_f[sj+1][si];
	}
	
	__syncthreads();
	
	for (int j = threadIdx.y; j < m_y; j += blockDim.y) { 
		int globalIdx = k * m_x * m_y + j * m_x + i;
		int sj = j + 4;
		df[globalIdx] = 
			( c_ay * ( s_f[sj+1][si] - s_f[sj-1][si] )
			+ c_by * ( s_f[sj+2][si] - s_f[sj-2][si] )
			+ c_cy * ( s_f[sj+3][si] - s_f[sj-3][si] )
			+ c_dy * ( s_f[sj+4][si] - s_f[sj-4][si] ) );
	}
}
						

// ------------
// z derivative
// ------------

__global__ void derivative_z(float *f, float *df)
{
	__shared__ float s_f[m_z+8][sPencils];
	
	int i  = blockIdx.x*blockDim.x + threadIdx.x;
	int j  = blockIdx.y;
	int k  = threadIdx.y;
	int si = threadIdx.x;
	int sk = k + 4; // halo offset

	int globalIdx = k * m_x * m_y + j * m_x + i;
	
	s_f[sk][si] = f[globalIdx];
	
	__syncthreads();
	
	if (k < 4) {
		s_f[sk-4][si]   = s_f[sk+m_z-5][si];
		s_f[sk+m_z][si] = s_f[sk+1][si];
	}
	
	__syncthreads();
	
	df[globalIdx] = 
		( c_az * ( s_f[sk+1][si] - s_f[sk-1][si] )
		+ c_bz * ( s_f[sk+2][si] - s_f[sk-2][si] )
		+ c_cz * ( s_f[sk+3][si] - s_f[sk-3][si] )
		+ c_dz * ( s_f[sk+4][si] - s_f[sk-4][si] ) );
}
	
__global__ void derivative_z_lPencils(float *f, float *df)
{
	__shared__ float s_f[m_z+8][lPencils];
	
	int i  = blockIdx.x*blockDim.x + threadIdx.x;
	int j  = blockIdx.y;
	int si = threadIdx.x;
	
	for (int k = threadIdx.y; k < m_z; k += blockDim.y) {
		int globalIdx = k * m_x * m_y + j * m_x + i;
		int sk = k + 4;
		s_f[sk][si] = f[globalIdx];
	}
	
	__syncthreads();
	
	int k = threadIdx.y + 4;
	if (k < 8) {
		s_f[k-4][si]  = s_f[k+m_z-5][si];
		s_f[k+m_z][si] = s_f[k+1][si];
	}
	
	__syncthreads();
	
	for (int k = threadIdx.y; k < m_z; k += blockDim.y) {
		int globalIdx = k * m_x * m_y + j * m_x + i;
		int sk = k +4;
		df[globalIdx] = 
			( c_az * ( s_f[sk+1][si] - s_f[sk-1][si] )
			+ c_bz * ( s_f[sk+2][si] - s_f[sk-2][si] )
			+ c_cz * ( s_f[sk+3][si] - s_f[sk-3][si] )
			+ c_dz * ( s_f[sk+4][si] - s_f[sk-4][si] ) );
	}
}

// Run the kernels for a given dimension.  One for sPencils, one for lPencils
void runTest(int dimension)
{
	void (*fpDeriv[2])(float*, float*);
	
	switch(dimension) {
		case 0:
			fpDeriv[0] = derivative_x;
			fpDeriv[1] = derivative_x_lPencils;
			break;
		case 1:
			fpDeriv[0] = derivative_y;
			fpDeriv[1] = derivative_y_lPencils;
			break;
		case 2:
			fpDeriv[0] = derivative_z;
			fpDeriv[1] = derivative_z_lPencils;
			break;
	}
	
	int sharedDims[3][2][2] = { m_x, sPencils,
								m_x, lPencils,
								sPencils, m_y, 
								lPencils, m_y,
								sPencils, m_z,
								lPencils, m_z };
				
	float f[m_x*m_y*m_z];
	float df[m_x*m_y*m_z];
	float sol[m_x*m_y*m_z];
	
	initInput(f, dimension);
	initSol(sol, dimension);
	
	// device arrays
	int bytes = m_x*m_y*m_z * sizeof(float);
	float *d_f, *d_df;
	HANDLE_ERROR( 
		cudaMalloc((void**)&d_f, bytes) );
	HANDLE_ERROR( 
		cudaMalloc((void**)&d_df, bytes) );
	
	const int nReps = 20;
	float milliseconds;
	cudaEvent_t startEvent, stopEvent;
	HANDLE_ERROR( 
		cudaEventCreate(&startEvent) );
	HANDLE_ERROR( 
		cudaEventCreate(&stopEvent) );
	
	double error, maxError;
	
	printf("%c derivatives\n\n", (char)(0x58 + dimension));
	
	for (int fp = 0; fp < 2; fp++) {
		HANDLE_ERROR( 
			cudaMemcpy(d_f, f, bytes, cudaMemcpyHostToDevice) );
		HANDLE_ERROR( 
			cudaMemset(d_df, 0, bytes) );
		
		fpDeriv[fp]<<<grid[dimension][fp], block[dimension][fp]>>>(d_f, d_df);
		HANDLE_ERROR( 
			cudaEventRecord(startEvent, 0) );
		for (int i = 0; i < nReps; i++)
			fpDeriv[fp]<<<grid[dimension][fp],block[dimension][fp]>>>(d_f, d_df);
			
		HANDLE_ERROR( 
			cudaEventRecord(stopEvent, 0) );
		HANDLE_ERROR( 
			cudaEventSynchronize(stopEvent) );
		HANDLE_ERROR( 
			cudaEventElapsedTime(&milliseconds, startEvent, stopEvent) );
		
		HANDLE_ERROR( 
			cudaMemcpy(df, d_df, bytes, cudaMemcpyDeviceToHost) );
		
		checkResults(error, maxError, sol, df);
		
		printf("  Using shared memory tile of %d x %d\n",
				sharedDims[dimension][fp][0], sharedDims[dimension][fp][1]);
		printf("   RMS error: %e\n", error);
		printf("   MAX error: %e\n", maxError);
		printf("   Average time (ms): %f\n", milliseconds / nReps);
		printf("   Average Bandwidth (GB/s): %f\n\n",
				2.f * 1e-6 * m_x * m_y * m_z * nReps * sizeof(float) / milliseconds);
	}
	
	HANDLE_ERROR( 
		cudaEventDestroy(startEvent) );
	HANDLE_ERROR( 
		cudaEventDestroy(stopEvent) );
	
	HANDLE_ERROR( 
		cudaFree(d_f) );
	HANDLE_ERROR( 
		cudaFree(d_df) );
}	



// This is the main host code for the finite difference
// example.  The kernels are contained in the derivative_m module

int main(void)
{
	// Print device and precision
	cudaDeviceProp prop;
	HANDLE_ERROR( 
		cudaGetDeviceProperties(&prop, 0) );
	printf("\nDevice name: %s\n", prop.name);
	printf("Compute Capability: %d.%d\n\n", prop.major, prop.minor);
	
	setDerivativeParameters(); // initialize
	
	runTest(0); // x derivative
	runTest(1); // y derivative
	runTest(2); // z derivative
	
	return 0;
}

