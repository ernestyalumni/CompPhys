/**
 * @file   : cg_eg4.cu
 * @brief  : Examples of using cooperative groups for partitioning groups, tiled partitions  
 * @details : cooperative groups for CUDA examples for partitioning groups, tiled partitions  
 *  Note; limitations due to 32-bit architecture of 
 * GeForce GTX 980 Ti that I'm using; please make a hardware donation (for a Titan V or GTX 1080 Ti) if you find this code useful!  
 * @author : Ernest Yeung <ernestyalumni@gmail.com>
 * @date   : 20170112      
 * @ref    : https://devblogs.nvidia.com/parallelforall/cooperative-groups/
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
/* 
 * COMPILATION TIP
 * nvcc cg_eg4.cu -o cg_eg4
 * 
 * */
#include <cooperative_groups.h>  

#include <stdio.h>  // printf

#include <iostream>

namespace cg = cooperative_groups;  

// demonstrates both modularity and partitioning og thread block into tiles of 32 threads
/** @fn partition_tile32
 * @brief demonstrates both modularity and partitioning og thread block into tiles of 32 threads
 * */
__device__ void partition_tile32(cg::thread_group & tg) {
	cg::thread_group tile32 = cg::tiled_partition(tg, 32);  
	cg::thread_group tile4 = cg::tiled_partition(tile32, 4); 

	if (tile4.thread_rank() ==0 ) {
		printf("Hello from tile4 rank 0: %d, tile32.thread_rank(): %d, tile32.size(): %d, tile4.size(): %d \n", 
			tg.thread_rank(), tile32.thread_rank(), tile32.size(), tile4.size() );
	}

//	dim3 tile32gidx = tile32.group_index(); // error: class "cooperative_groups::__v1::thread_group" has no member "group_index"
//	dim3 tile32tidx = tile32.thread_index();  // error: class "cooperative_groups::__v1::thread_group" has no member "thread_index"
//	dim3 tggidx = tg.group_index(); // same error
//	dim3 tgtidx = tg.thread_index(); 
//	printf("group_index: tile32gidx.x: %d  .y: %d  .z: %d \n", tile32gidx.x,tile32gidx.y,tile32gidx.z); 
//	printf("thread_index: tile32tidx.x: %d  .y: %d  .z: %d\n", tile32tidx.x,tile32tidx.y,tile32tidx.z); 
//	printf("group_index: tggidx.x: %d  .y: %d  .z: %d \n", tggidx.x,tggidx.y,tggidx.z); 
//	printf("thread_index: tgtidx.x: %d  .y: %d  .z: %d\n", tgtidx.x,tgtidx.y,tgtidx.z); 
//	printf(" %d ", tg.thread_rank());

}




__global__ void explore_tiled_blocks_kernel()
{
	// get this thread block; Handle to thread block group  
	auto cgtb = cg::this_thread_block(); 
	partition_tile32( cgtb ); 

/*	dim3 tggidx = cgtb.group_index(); // same error
	dim3 tgtidx = cgtb.thread_index(); 
	printf("group_index: cgtb.x: %d  .y: %d  .z: %d \n", tggidx.x,tggidx.y,tggidx.z); 
	printf("thread_index: cgtb.x: %d  .y: %d  .z: %d\n", tgtidx.x,tgtidx.y,tgtidx.z); 
*/
};

/** @fn explore_tiled_blocks
 * @brief "driver" function for explore_tiled_blocks_kernel 
 */
void explore_tiled_blocks(const dim3 M_is,const size_t Lx, const size_t Ly=0 ) {
	/* ************************* */
	/* thread block, grid dims.  */
	/* ************************* */
	unsigned int Nx = (Lx + M_is.x - 1)/ M_is.x; // number of (thread) blocks in the x-direction  
	unsigned int Ny = 1; 
	if (Ly != 0) // check if we need to calculate Ny for this problem, or not
	{
		Ny = (Ly + M_is.y - 1)/ M_is.y; 
	}
//	int sharedBytes = blockSize * sizeof(int); 
	dim3 N_is { Nx,Ny }; 

	// run the kernel
	if (Ly == 0) {
		explore_tiled_blocks_kernel<<<Nx, M_is.x>>>(); 
//		cudaLaunchCooperativeKernel((void*) explore_tiled_blocks_kernel, N_is, M_is, nullptr) ;
	
	} else {
		explore_tiled_blocks_kernel<<<N_is,M_is>>>();  
//		cudaLaunchCooperativeKernel((void*) explore_tiled_blocks_kernel, N_is, M_is, nullptr) ;
	}
	cudaDeviceSynchronize(); // https://stackoverflow.com/questions/15669841/cuda-hello-world-printf-not-working-even-with-arch-sm-20

};

__global__ void explore_2dt_blocks_kernel()
{
	// get this thread block; Handle to thread block group  
	auto cgtb = cg::this_thread_block(); 

	// compare with before
	printf("threadIdx.x: %d, .y: %d, blockDim.x: %d, .y: %d, blockIdx.x: %d, .y: %d, gridDim.x: %d, gridDim.y: %d\n", 
		threadIdx.x,threadIdx.y,blockDim.x,blockDim.y,blockIdx.x,blockIdx.y,gridDim.x,gridDim.y); 

	// compare with now 
	printf("cgtb.size(): %d, cgtb.thread_rank(): %d\n", cgtb.size(), cgtb.thread_rank());

//	cta.is_valid(); // error: class "cooperative_groups::__v1::thread_block" has no member "is_valid"

	dim3 cgtb_gidx = cgtb.group_index();
	dim3 cgtb_tidx = cgtb.thread_index(); 
	printf("group_index: .x: %d  .y: %d  .z: %d \n", cgtb_gidx.x,cgtb_gidx.y,cgtb_gidx.z); 
	printf("thread_index: .x: %d  .y: %d  .z: %d\n", cgtb_tidx.x,cgtb_tidx.y,cgtb_tidx.z); 


};

/** @fn explore_2dt_blocks 
 * @brief "driver" function for explore_2dt_blocks_kernel 
 * */
void explore_2dt_blocks(const dim3 M_is,const size_t Lx, const size_t Ly) {
	/* ************************* */
	/* thread block, grid dims.  */
	/* ************************* */
	unsigned int Nx = (Lx + M_is.x - 1)/ M_is.x; // number of (thread) blocks in the x-direction  
	unsigned int Ny = (Ly + M_is.y - 1)/ M_is.y; 

	const dim3 N_is { Nx,Ny }; 

	// run the kernel
	cudaLaunchCooperativeKernel( (void*) explore_2dt_blocks_kernel, N_is, M_is, nullptr  );

	// cudaDeviceSynchronize() needed for printf
	cudaDeviceSynchronize(); // https://stackoverflow.com/questions/15669841/cuda-hello-world-printf-not-working-even-with-arch-sm-20
	
};

int main(int argc, char* argv[]) 
{
	dim3 M_is { 32 }; 
	explore_tiled_blocks( M_is, 2*32 ,0);

	M_is = {2,2};
	explore_2dt_blocks( M_is, 4,4 );
}
