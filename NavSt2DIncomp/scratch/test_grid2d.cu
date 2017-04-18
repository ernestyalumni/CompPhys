/**
 * @file   : R2grid.h
 * @brief  : R2 under discretization (discretize functor) to a (staggered) grid
 * uses CUDA Unified Memory (Management)
 * @author : Ernest Yeung	ernestyalumni@gmail.com
 * @date   : 20170414
 * @ref : cf. http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared
 * If you find this code useful, feel free to donate directly and easily at this direct PayPal link: 
 * 
 * https://www.paypal.com/cgi-bin/webscr?cmd=_donations&business=ernestsaveschristmas%2bpaypal%40gmail%2ecom&lc=US&item_name=ernestyalumni&currency_code=USD&bn=PP%2dDonationsBF%3abtn_donateCC_LG%2egif%3aNonHosted 
 * 
 * which won't go through a 3rd. party such as indiegogo, kickstarter, patreon.  
 * Otherwise, I receive emails and messages on how all my (free) material on 
 * physics, math, and engineering have helped students with their studies, 
 * and I know what it's like to not have money as a student, but love physics 
 * (or math, sciences, etc.), so I am committed to keeping all my material 
 * open-source and free, whether or not 
 * sufficiently crowdfunded, under the open-source MIT license: 
 * 	feel free to copy, edit, paste, make your own versions, share, use as you wish.  
 *  Just don't be an asshole and not give credit where credit is due.  
 * Peace out, never give up! -EY
 * 
 * */
 /**
  * COMPILATION TIP(s)
  * 
  * nvcc test_grid2d.cu ./physlib/R2grid.cu -o test_grid2d.exe
  * but for __managed__ variables, you'll need to specify gpu-architecture or gpu compute capability:
  * nvcc -arch='sm_52' test_grid2d.cu ./physlib/R2grid.cu -o test_grid2d.exe
  * */
#include <iostream>

#include "./commonlib/checkerror.h"
#include "./physlib/R2grid.h"

dim3 Ld_global(4,6,1);

Grid2d Grid2d_global(Ld_global);
Grid2d_basic grid2d_global_basic(Ld_global);

__device__ __managed__ Bundles<(4+2)*(6+2)> bundles;

__device__ __managed__ Bundles<(16)*(8)> bundles_test;

// cf.  http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared
extern __shared__ float sh_F[];
__global__ void compute_F(float *u, float* v, float* F,
	const int Lx, const int Ly) {
	int W=3;
	int RAD= W/2;
	int k_x=threadIdx.x +blockDim.x*blockIdx.x;
	int k_y=threadIdx.y+blockDim.y*blockIdx.y;
	
	const int S_x = static_cast<int>(blockDim.x+2*RAD);
	const int S_y = static_cast<int>(blockDim.y+2*RAD);
		
	const int s_x = threadIdx.x + RAD;
	const int s_y = threadIdx.x + RAD;
		
	int l_x=0;
	int l_y=0;
	
//	float* sh_u = &sh_F[S_x*S_y];  // this works as well,no an illegal memory access was encountered cudaFree(u) 
//	float* sh_v = &sh_u[S_x*S_y];	// this works as well, no an illegal memory access was encountered cudaFree(u)

//	float* sh_u = &sh_F[S_x][S_y];  // error: expression must have pointer-to-object type
	
//	float* sh_u = sh_F[S_x][S_y]; // error: constant value is not known
//	float* sh_u = sh_F[S_x*S_y];
//	float* sh_v = sh_u[S_x*S_y];

//	float* sh_u = &sh_F; // error: a value of type "float (*)[]" cannot be used to initialize an entity of type "float *
//	float* sh_v = &sh_u[S_x*S_y]; // error: a value of type "float (*)[]" cannot be used to initialize an entity of type "float *

	float* sh_u = sh_F ;
	float* sh_v = sh_u + S_x*S_y ; 


	for (int nu_x=0; nu_x < S_x; nu_x++) { 
		for (int nu_y=0;nu_y<S_y; nu_y++) {
			sh_u[nu_x + S_x*nu_y] = nu_x + nu_y*0.5f ; 
			sh_v[nu_x + S_x*nu_y] = nu_x * 1.2f + nu_y ; } 
			
	}
}

extern __shared__ float sh_p[];
__global__ void compute_FLAG(float *u, float* v, float* F,
	const int Lx, const int Ly) {
	int W=3;
	int RAD= W/2;
	int k_x=threadIdx.x +blockDim.x*blockIdx.x;
	int k_y=threadIdx.y+blockDim.y*blockIdx.y;
	
	const int S_x = static_cast<int>(blockDim.x+2*RAD);
	const int S_y = static_cast<int>(blockDim.y+2*RAD);
		
	const int s_x = threadIdx.x + RAD;
	const int s_y = threadIdx.x + RAD;
		
	int l_x=0;
	int l_y=0;

	float* sh_u = sh_p ;
	int* sh_FLAG = (int*) (sh_u + S_x*S_y ); 

	for (int nu_x=0; nu_x < S_x; nu_x++) { 
		for (int nu_y=0;nu_y<S_y; nu_y++) {
			sh_u[nu_x + S_x*nu_y] = nu_x + nu_y*0.5f ; 
			sh_FLAG[nu_x + S_x*nu_y] = nu_x * 2 + nu_y ; } 
			
	}
	
	__syncthreads();	

}


int main(int argc, char *argv[])
{
	dim3 Ld(4,6,1);
	Grid2d grid2d(Ld);
	
	std::cout << "grid2d_local.Ld "<< grid2d.Ld.x << grid2d.Ld.y << grid2d.Ld.z << std::endl;
	std::cout << "grid2d.staggered_Ld "<< grid2d.staggered_Ld.x << grid2d.staggered_Ld.y << grid2d.staggered_Ld.z << std::endl;
	std::cout << "grid2d.NFLAT() " << grid2d.NFLAT() << std::endl;
	std::cout << "grid2d.staggered_SIZE() " << grid2d.staggered_SIZE() << std::endl;
	std::cout << "grid2d.flatten(2,3) " << grid2d.flatten(2,3) << std::endl;
	std::cout << "grid2d.staggered_flatten(2,3) " << grid2d.staggered_flatten(2,3) << std::endl;
	
	// works globally
	std::cout << "works globally : " << std::endl;
	std::cout << "Grid2d_global.Ld "<< Grid2d_global.Ld.x << Grid2d_global.Ld.y << Grid2d_global.Ld.z << std::endl;
	std::cout << "Grid2d_global.staggered_Ld "<< Grid2d_global.staggered_Ld.x << Grid2d_global.staggered_Ld.y << Grid2d_global.staggered_Ld.z << std::endl;
	std::cout << "Grid2d_global.NFLAT() " << Grid2d_global.NFLAT() << std::endl;
	std::cout << "Grid2d_global.staggered_SIZE() " << Grid2d_global.staggered_SIZE() << std::endl;
	std::cout << "Grid2d_global.flatten(2,3) " << Grid2d_global.flatten(2,3) << std::endl;
	std::cout << "Grid2d_global.staggered_flatten(2,3) " << Grid2d_global.staggered_flatten(2,3) << std::endl;
	
	for (int idx = 0; idx < 16*8 ; idx++) { 
		bundles_test.u[idx] = ((float) idx);
		bundles_test.v[idx] = ((float) idx) + 2.f;
		bundles_test.F[idx] = ((float) idx)*0.25f;
	}	
	
	dim3 M_i(4,2);
	dim3 griddims((16-2 +M_i.x-1)/M_i.x,(8-2+M_i.y-1)/M_i.y)   ;
	int smSz = (M_i.x+2)*(M_i.y+2)*sizeof(float)*2 ; // 2 float arrays
	compute_F<<< griddims, M_i, smSz>>>( bundles_test.u, 
		bundles_test.v, bundles_test.F, 14, 6 );
	
	int smSz_FLAG = (M_i.x+2)*(M_i.y+2)*sizeof(float) + (M_i.x+2)*(M_i.y+2)*sizeof(int); // 1 float array, 1 int array

	compute_FLAG<<< griddims, M_i, smSz_FLAG >>>( bundles_test.u, 
		bundles_test.v, bundles_test.F, 14, 6 );
	
	
	return 0;	
}

