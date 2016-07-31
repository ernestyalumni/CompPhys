/* heat_2d.cu
 * 2-dim. Laplace eq. (heat eq.) by finite difference with shared memory
 * Ernest Yeung  ernestyalumni@gmail.com
 * 20160625
 */
#include "heat_2d.h"

#define RAD 1 // radius of the stencil; helps to deal with "boundary conditions" at (thread) block's ends

__constant__ float dev_Deltat[1];

__constant__ float dev_heat_params[2];



int blocksNeeded( int N_i, int M_i) { return (N_i+M_i-1)/M_i; }

__device__ unsigned char clip(int n) { return n > 255 ? 255 : (n < 0 ? 0 : n);}

__device__ int idxClip( int idx, int idxMax) {
	return idx > (idxMax - 1) ? (idxMax - 1): (idx < 0 ? 0 : idx);
}

__device__ int flatten(int col, int row, int width, int height) {
	return idxClip(col, width) + idxClip(row,height)*width;
}

__global__ void resetKernel(float *d_temp, BC bc) {
	const int col = blockIdx.x*blockDim.x + threadIdx.x;
	const int row = blockIdx.y*blockDim.y + threadIdx.y;
	if ((col >= dev_Ld[0]) || (row >= dev_Ld[1])) return;
	d_temp[row*dev_Ld[0] + col] = bc.t_a;
}


__global__ void tempKernel(float *d_temp, BC bc) {
	constexpr int NUS = 1;
	constexpr int radius = NUS;
	
	extern __shared__ float s_in[];
	// global indices
	const int k_x = threadIdx.x + blockDim.x * blockIdx.x;
	const int k_y = threadIdx.y + blockDim.y * blockIdx.y;
	if ((k_x >= dev_Ld[0] ) || (k_y >= dev_Ld[1] )) return;
	const int k = flatten(k_x, k_y, dev_Ld[0], dev_Ld[1]);
	// local width and height
	const int2 S = { static_cast<int>(blockDim.x + 2 * radius), 
						static_cast<int>(blockDim.y + 2 * radius) };

	// local indices
	const int s_x = threadIdx.x + radius;
	const int s_y = threadIdx.y + radius;
	const int s_k = flatten(s_x, s_y, S.x, S.y);
	// assign default color values for d_out (black)

	// Load regular cells
	s_in[s_k] = d_temp[k];
	// Load halo cells
	if (threadIdx.x < radius ) {
		s_in[flatten(s_x - radius, s_y, S.x, S.y)] = 
			d_temp[flatten(k_x - radius, k_y, dev_Ld[0], dev_Ld[1])];
		s_in[flatten(s_x + blockDim.x, s_y, S.x, S.y)] = 
			d_temp[flatten(k_x + blockDim.x, k_y, dev_Ld[0], dev_Ld[1])];
	}
	if (threadIdx.y < radius) {
		s_in[flatten(s_x, s_y - radius, S.x, S.y)] = 
			d_temp[flatten(k_x, k_y - radius, dev_Ld[0], dev_Ld[1])];
		s_in[flatten(s_x, s_y + blockDim.y, S.x, S.y)] = 
			d_temp[flatten(k_x, k_y + blockDim.y, dev_Ld[0], dev_Ld[1])];
	}
	
	// Calculate squared distance from pipe center
	float dSq = ((k_x - bc.x)*(k_x - bc.x) + (k_y - bc.y)*(k_y - bc.y));
	// If inside pipe, set temp to t_s and return
	if (dSq < bc.rad*bc.rad) {
		d_temp[k] = bc.t_s;
		return;
	}
	// If outside plate, set temp to t_a and return
	if ((k_x == 0 ) || (k_x == dev_Ld[0] - 1) || (k_y == 0 ) ||
		(k_x + k_y < bc.chamfer) || (k_x - k_y > dev_Ld[0] - bc.chamfer)) {
			d_temp[k] = bc.t_a;
			return;
	}
	// If point is below ground, set temp to t_g and return
	if (k_y == dev_Ld[1] - 1) {
		d_temp[k] = bc.t_g;
		return;
	}
	__syncthreads();
	// For all the remaining points, find temperature.
	
	float2 stencil[NUS][2] ; 
	
	const float centerval { s_in[flatten(s_x,s_y,S.x,S.y)] };
	
	for (int nu = 0; nu < NUS; ++nu) {
		stencil[nu][0].x = s_in[flatten(s_x-(nu+1),s_y,S.x,S.y)] ; 
		stencil[nu][1].x = s_in[flatten(s_x+(nu+1),s_y,S.x,S.y)] ; 
		stencil[nu][0].y = s_in[flatten(s_x,s_y-(nu+1),S.x,S.y)] ; 
		stencil[nu][1].y = s_in[flatten(s_x,s_y+(nu+1),S.x,S.y)] ; 
	}

	float tempval { dev_lap1( centerval, stencil ) };

	__syncthreads();

	d_temp[k] += dev_Deltat[0]*(dev_heat_params[0]/dev_heat_params[1])*tempval;

}

__global__ void tempKernel2(float *d_temp, BC bc) {
	constexpr int NUS = 2;
	constexpr int radius = NUS;

	extern __shared__ float s_in[];
	// global indices
	const int k_x = threadIdx.x + blockDim.x * blockIdx.x;
	const int k_y = threadIdx.y + blockDim.y * blockIdx.y;
	if ((k_x >= dev_Ld[0] ) || (k_y >= dev_Ld[1] )) return;
	const int k = flatten(k_x, k_y, dev_Ld[0], dev_Ld[1]);
	// local width and height
	const int2 S { static_cast<int>(blockDim.x + 2 * radius) , 
					static_cast<int>(blockDim.y + 2 * radius) } ; 

	// local indices
	const int s_x = threadIdx.x + radius;
	const int s_y = threadIdx.y + radius;
	const int s_k = flatten(s_x, s_y, S.x, S.y);
	// assign default color values for d_out (black)

	// Load regular cells
	s_in[s_k] = d_temp[k];
	// Load halo cells
	if (threadIdx.x < radius ) {
		s_in[flatten(s_x - radius, s_y, S.x, S.y)] = 
			d_temp[flatten(k_x - radius, k_y, dev_Ld[0], dev_Ld[1])];
		s_in[flatten(s_x + blockDim.x, s_y, S.x, S.y)] = 
			d_temp[flatten(k_x + blockDim.x, k_y, dev_Ld[0], dev_Ld[1])];
	}
	if (threadIdx.y < radius) {
		s_in[flatten(s_x, s_y - radius, S.x, S.y)] = 
			d_temp[flatten(k_x, k_y - radius, dev_Ld[0], dev_Ld[1])];
		s_in[flatten(s_x, s_y + blockDim.y, S.x, S.y)] = 
			d_temp[flatten(k_x, k_y + blockDim.y, dev_Ld[0], dev_Ld[1])];
	}
	
	// Calculate squared distance from pipe center
	float dSq = ((k_x - bc.x)*(k_x - bc.x) + (k_y - bc.y)*(k_y - bc.y));
	// If inside pipe, set temp to t_s and return
	if (dSq < bc.rad*bc.rad) {
		d_temp[k] = bc.t_s;
		return;
	}
	// If outside plate, set temp to t_a and return
	if ((k_x == 0 ) || (k_x == dev_Ld[0] - 1) || (k_y == 0 ) ||
		(k_x + k_y < bc.chamfer) || (k_x - k_y > dev_Ld[0] - bc.chamfer)) {
			d_temp[k] = bc.t_a;
			return;
	}
	// If point is below ground, set temp to t_g and return
	if (k_y == dev_Ld[1] - 1) {
		d_temp[k] = bc.t_g;
		return;
	}
	__syncthreads();
	// For all the remaining points, find temperature.
	
	float2 stencil[NUS][2] ; 
	
	const float centerval { s_in[flatten(s_x,s_y,S.x,S.y)] };
	
	for (int nu = 0; nu < NUS; ++nu) {
		stencil[nu][0].x = s_in[flatten(s_x-(nu+1),s_y,S.x,S.y)] ; 
		stencil[nu][1].x = s_in[flatten(s_x+(nu+1),s_y,S.x,S.y)] ; 
		stencil[nu][0].y = s_in[flatten(s_x,s_y-(nu+1),S.x,S.y)] ; 
		stencil[nu][1].y = s_in[flatten(s_x,s_y+(nu+1),S.x,S.y)] ; 
	}

	float tempval { dev_lap2( centerval, stencil ) };

	__syncthreads();

	d_temp[k] += dev_Deltat[0]*(dev_heat_params[0]/dev_heat_params[1])*tempval;

}

__global__ void tempKernel3(float *d_temp, BC bc) {
	constexpr int NUS = 3;
	constexpr int radius = NUS;

	extern __shared__ float s_in[];
	// global indices
	const int k_x = threadIdx.x + blockDim.x * blockIdx.x;
	const int k_y = threadIdx.y + blockDim.y * blockIdx.y;
	if ((k_x >= dev_Ld[0] ) || (k_y >= dev_Ld[1] )) return;
	const int k = flatten(k_x, k_y, dev_Ld[0], dev_Ld[1]);
	// local width and height
	const int2 S { static_cast<int>(blockDim.x + 2 * radius) , 
					static_cast<int>(blockDim.y + 2 * radius) } ; 

	// local indices
	const int s_x = threadIdx.x + radius;
	const int s_y = threadIdx.y + radius;
	const int s_k = flatten(s_x, s_y, S.x, S.y);
	// assign default color values for d_out (black)

	// Load regular cells
	s_in[s_k] = d_temp[k];
	// Load halo cells
	if (threadIdx.x < radius ) {
		s_in[flatten(s_x - radius, s_y, S.x, S.y)] = 
			d_temp[flatten(k_x - radius, k_y, dev_Ld[0], dev_Ld[1])];
		s_in[flatten(s_x + blockDim.x, s_y, S.x, S.y)] = 
			d_temp[flatten(k_x + blockDim.x, k_y, dev_Ld[0], dev_Ld[1])];
	}
	if (threadIdx.y < radius) {
		s_in[flatten(s_x, s_y - radius, S.x, S.y)] = 
			d_temp[flatten(k_x, k_y - radius, dev_Ld[0], dev_Ld[1])];
		s_in[flatten(s_x, s_y + blockDim.y, S.x, S.y)] = 
			d_temp[flatten(k_x, k_y + blockDim.y, dev_Ld[0], dev_Ld[1])];
	}
	
	// Calculate squared distance from pipe center
	float dSq = ((k_x - bc.x)*(k_x - bc.x) + (k_y - bc.y)*(k_y - bc.y));
	// If inside pipe, set temp to t_s and return
	if (dSq < bc.rad*bc.rad) {
		d_temp[k] = bc.t_s;
		return;
	}
	// If outside plate, set temp to t_a and return
	if ((k_x == 0 ) || (k_x == dev_Ld[0] - 1) || (k_y == 0 ) ||
		(k_x + k_y < bc.chamfer) || (k_x - k_y > dev_Ld[0] - bc.chamfer)) {
			d_temp[k] = bc.t_a;
			return;
	}
	// If point is below ground, set temp to t_g and return
	if (k_y == dev_Ld[1] - 1) {
		d_temp[k] = bc.t_g;
		return;
	}
	__syncthreads();
	// For all the remaining points, find temperature.
	
	float2 stencil[NUS][2] ; 
	
	const float centerval { s_in[flatten(s_x,s_y,S.x,S.y)] };
	
	for (int nu = 0; nu < NUS; ++nu) {
		stencil[nu][0].x = s_in[flatten(s_x-(nu+1),s_y,S.x,S.y)] ; 
		stencil[nu][1].x = s_in[flatten(s_x+(nu+1),s_y,S.x,S.y)] ; 
		stencil[nu][0].y = s_in[flatten(s_x,s_y-(nu+1),S.x,S.y)] ; 
		stencil[nu][1].y = s_in[flatten(s_x,s_y+(nu+1),S.x,S.y)] ; 
	}

	float tempval { dev_lap3( centerval, stencil ) };

	__syncthreads();

	d_temp[k] += dev_Deltat[0]*(dev_heat_params[0]/dev_heat_params[1])*tempval;

}

__global__ void tempKernel4(float *d_temp, BC bc) {
	constexpr int NUS = 4;
	constexpr int radius = NUS;

	extern __shared__ float s_in[];
	// global indices
	const int k_x = threadIdx.x + blockDim.x * blockIdx.x;
	const int k_y = threadIdx.y + blockDim.y * blockIdx.y;
	if ((k_x >= dev_Ld[0] ) || (k_y >= dev_Ld[1] )) return;
	const int k = flatten(k_x, k_y, dev_Ld[0], dev_Ld[1]);
	// local width and height
	const int2 S { static_cast<int>(blockDim.x + 2 * radius) , 
					static_cast<int>(blockDim.y + 2 * radius) } ; 

	// local indices
	const int s_x = threadIdx.x + radius;
	const int s_y = threadIdx.y + radius;
	const int s_k = flatten(s_x, s_y, S.x, S.y);
	// assign default color values for d_out (black)

	// Load regular cells
	s_in[s_k] = d_temp[k];
	// Load halo cells
	if (threadIdx.x < radius ) {
		s_in[flatten(s_x - radius, s_y, S.x, S.y)] = 
			d_temp[flatten(k_x - radius, k_y, dev_Ld[0], dev_Ld[1])];
		s_in[flatten(s_x + blockDim.x, s_y, S.x, S.y)] = 
			d_temp[flatten(k_x + blockDim.x, k_y, dev_Ld[0], dev_Ld[1])];
	}
	if (threadIdx.y < radius) {
		s_in[flatten(s_x, s_y - radius, S.x, S.y)] = 
			d_temp[flatten(k_x, k_y - radius, dev_Ld[0], dev_Ld[1])];
		s_in[flatten(s_x, s_y + blockDim.y, S.x, S.y)] = 
			d_temp[flatten(k_x, k_y + blockDim.y, dev_Ld[0], dev_Ld[1])];
	}
	
	// Calculate squared distance from pipe center
	float dSq = ((k_x - bc.x)*(k_x - bc.x) + (k_y - bc.y)*(k_y - bc.y));
	// If inside pipe, set temp to t_s and return
	if (dSq < bc.rad*bc.rad) {
		d_temp[k] = bc.t_s;
		return;
	}
	// If outside plate, set temp to t_a and return
	if ((k_x == 0 ) || (k_x == dev_Ld[0] - 1) || (k_y == 0 ) ||
		(k_x + k_y < bc.chamfer) || (k_x - k_y > dev_Ld[0] - bc.chamfer)) {
			d_temp[k] = bc.t_a;
			return;
	}
	// If point is below ground, set temp to t_g and return
	if (k_y == dev_Ld[1] - 1) {
		d_temp[k] = bc.t_g;
		return;
	}
	__syncthreads();
	// For all the remaining points, find temperature.
	
	float2 stencil[NUS][2] ; 
	
	const float centerval { s_in[flatten(s_x,s_y,S.x,S.y)] };
	
	for (int nu = 0; nu < NUS; ++nu) {
		stencil[nu][0].x = s_in[flatten(s_x-(nu+1),s_y,S.x,S.y)] ; 
		stencil[nu][1].x = s_in[flatten(s_x+(nu+1),s_y,S.x,S.y)] ; 
		stencil[nu][0].y = s_in[flatten(s_x,s_y-(nu+1),S.x,S.y)] ; 
		stencil[nu][1].y = s_in[flatten(s_x,s_y+(nu+1),S.x,S.y)] ; 
	}

	float tempval { dev_lap4( centerval, stencil ) };

	__syncthreads();

	d_temp[k] += dev_Deltat[0]*(dev_heat_params[0]/dev_heat_params[1])*tempval;

}



__global__ void float_to_char( uchar4* dev_out, const float* outSrc) {
	const int k_x = threadIdx.x + blockDim.x * blockIdx.x;
	const int k_y = threadIdx.y + blockDim.y * blockIdx.y;
	
	const int k   = k_x + k_y * blockDim.x*gridDim.x ; 

	dev_out[k].x = 0;
	dev_out[k].z = 0;
	dev_out[k].y = 0;
	dev_out[k].w = 255;


	const unsigned char intensity = clip((int) outSrc[k] ) ;
	dev_out[k].x = intensity ;       // higher temp -> more red
	dev_out[k].z = 255 - intensity ; // lower temp -> more blue
	
}


void kernelLauncher(uchar4 *d_out, float *d_temp, int w, int h, BC bc, dim3 M_in) {
	const dim3 gridSize(blocksNeeded(w, M_in.x), blocksNeeded(h, M_in.y));
	const size_t smSz = (M_in.x + 2 * RAD)*(M_in.y + 2 * RAD)*sizeof(float);

	tempKernel<<<gridSize, M_in, smSz>>>(d_temp, bc);

	float_to_char<<<gridSize,M_in>>>(d_out, d_temp) ; 
}

void kernelLauncher2(uchar4 *d_out, float *d_temp, int w, int h, BC bc, dim3 M_in) {
	constexpr int radius { 2 };
	
	const dim3 gridSize(blocksNeeded(w, M_in.x), blocksNeeded(h, M_in.y));
	const size_t smSz = (M_in.x + 2 * radius)*(M_in.y + 2 * radius)*sizeof(float);

	tempKernel2<<<gridSize, M_in, smSz>>>(d_temp, bc);

	float_to_char<<<gridSize,M_in>>>(d_out, d_temp) ; 
}

void kernelLauncher3(uchar4 *d_out, float *d_temp, int w, int h, BC bc, dim3 M_in) {
	constexpr int radius { 3 };
	
	const dim3 gridSize(blocksNeeded(w, M_in.x), blocksNeeded(h, M_in.y));
	const size_t smSz = (M_in.x + 2 * radius)*(M_in.y + 2 * radius)*sizeof(float);

	tempKernel3<<<gridSize, M_in, smSz>>>(d_temp, bc);

	float_to_char<<<gridSize,M_in>>>(d_out, d_temp) ; 
}

void kernelLauncher4(uchar4 *d_out, float *d_temp, int w, int h, BC bc, dim3 M_in) {
	constexpr int radius { 4 };
	
	const dim3 gridSize(blocksNeeded(w, M_in.x), blocksNeeded(h, M_in.y));
	const size_t smSz = (M_in.x + 2 * radius)*(M_in.y + 2 * radius)*sizeof(float);

	tempKernel4<<<gridSize, M_in, smSz>>>(d_temp, bc);

	float_to_char<<<gridSize,M_in>>>(d_out, d_temp) ; 
}

void resetTemperature(float *d_temp, int w, int h, BC bc, dim3 M_in) {
	const dim3 gridSize( blocksNeeded(w, M_in.x), blocksNeeded( h, M_in.y));

	resetKernel<<<gridSize, M_in>>>(d_temp,bc);
}


	
	
	
			
		
		
