/* cf. Jason Sanders, Edward Kandrot. CUDA by Example: An Introduction to General-Purpose GPU Programming */
/* 
** Chapter 8 Graphics Interoperability
** 8.4 Heat Transfer with Graphics Interop
*/

#include <cmath> /* acos */

#include "common/anim_color.h"
#include "common/errors.h"
#include "common/gpu_anim.h"


#define DIM 1024
#define PI acos(-1)
#define MAX_TEMP 1.0f
#define MIN_TEMP 0.0001f
#define SPEED 0.25f

/* texture references must be declared globally at file scope
 * declare 2-dim. textures */
// These exist on the GPU side
texture <float,2> texConstSrc;
texture <float,2> texIn;
texture <float,2> texOut; 

/* since copy_const_kernel() kernel reads from our buffer that holds heater positions and temperatures,
 * modify in order to read through texture memory instead of global memory
 * cf. pp. 129 of Sanders and Kandrot  */
// originally
// __global__ void copy_const_kernel( float *iptr, const float *cptr) {
__global__ void copy_const_kernel( float *iptr) {
	// map from threadIdx/blockIdx to pixel position
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	// simply use x,y to address constant source 
	float c = tex2D(texConstSrc,x,y);

// Notice that copy is performed only if cell in constant grid is nonzero
// We do this to preserve any values computed in previous time step within 
// cells that don't contain heaters	
	if (c != 0)
		iptr[offset] = c;
}


/* when reading from textures in the kernel (GPU, device), 
 * we need to use special functions to instruct GPU to route our requests 
 * through texture unit and not through standard global memory 
 *
 * no longer need to use linearized offset to compute top,left,right,bottom
 * we can use x and y directly to address texture
 */
__global__ void blend_kernel( float *dst, bool dstOut ){
	// map from threadIdx/blockIdx to pixel position
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;
	
	/* tex2D looks like a function, but it's a compiler intrinsic 
	 * AND compiler needs to know at compile time which textures tex2D
	 * should be sampling, and since
	 * texture references must be declared globally at file scope,
	 * hence dstOut flag
	*/
	/* tex2D boundary conditions
	 * Furthermore, if 1 of x or y is less than 0, tex2D() return value 0
	 * if 1 of x or y is greater than width, tex2D() return value at width 1 wraps around
	 */
	float t, l, c, r, b;
	if (dstOut) {
		t = tex2D(texIn,x,y-1);
		l = tex2D(texIn,x-1,y);
		c = tex2D(texIn,x,y);
		r = tex2D(texIn,x+1,y);
		b = tex2D(texIn,x,y+1);
	}
	else {
		t = tex2D(texOut,x,y-1);
		l = tex2D(texOut,x-1,y);
		c = tex2D(texOut,x,y);
		r = tex2D(texOut,x+1,y);
		b = tex2D(texOut,x,y+1);
	}
	
	// new update step
	dst[offset] = c + SPEED * (t + b + r + l - 4*c);	
}

// globals needed by the update routine
struct DataBlock {
	unsigned char  *output_bitmap;
	float          *dev_inSrc;
	float          *dev_outSrc;
	float          *dev_constSrc;
	GPUAnimBitmap  *bitmap;
	cudaEvent_t    start, stop;
	float          totalTime;
	float          frames;
};	
	
void anim_gpu( uchar4* outputBitmap, DataBlock *d, int ticks) {
	HANDLE_ERROR( cudaEventRecord( d->start, 0) );
	dim3    blocks(DIM/16,DIM/16);
	dim3    threads(16,16);

/* since blend_kernel() changed to accept a boolean flag that 
 * switches the buffers between input and output, 
 * rather than swap buffers, set dstOut = !dstOut to toggle flag after each series of calls
 */
	// "new" dstOut
	volatile bool dstOut = true;
	
	for (int i = 0; i < 90; i++) {
		// pointers to float in, out are new
		float *in, *out;
		if (dstOut) {
			in  = d->dev_inSrc;
			out = d->dev_outSrc;
		}
		else {
			out = d->dev_inSrc;
			in  = d->dev_outSrc;
		}

		/* original code; no need to swap buffers, only toggle fla
		copy_const_kernel<<<blocks,threads>>>(  d->dev_inSrc,
												d->dev_constSrc );
		blend_kernel<<<blocks,threads>>>(   d->dev_outSrc,
											d->dev_inSrc );
		swap( d->dev_inSrc, d->dev_outSrc );
		*/
		copy_const_kernel<<<blocks,threads>>>( in );
		blend_kernel<<<blocks,threads>>>( out, dstOut );
		dstOut = !dstOut;				
	}
	float_to_color<<<blocks,threads>>>( outputBitmap,
										d->dev_inSrc );
						
	HANDLE_ERROR(
		cudaEventRecord( d->stop, 0) );
	HANDLE_ERROR(
		cudaEventSynchronize( d->stop ));
	float elapsedTime;
	HANDLE_ERROR(
		cudaEventElapsedTime(   &elapsedTime,
								d->start, d->stop) );
	d->totalTime += elapsedTime;
	++d->frames;
	printf( "Average Time per frame:  %3.1f ms\n",
			d->totalTime/d->frames );
}

int main( void ) {
	DataBlock  data;
	GPUAnimBitmap bitmap( DIM, DIM, &data );
	data.bitmap = &bitmap;
	data.totalTime = 0;
	data.frames = 0;
	HANDLE_ERROR(
					cudaEventCreate( &data.start ));
	HANDLE_ERROR(
					cudaEventCreate( &data.stop  ));
	
	HANDLE_ERROR(
					cudaMalloc( (void**)&data.output_bitmap, bitmap.image_size() ));
					
	// assume float == 4 chars in size (i.e., rgba )
	HANDLE_ERROR(
		cudaMalloc( (void**)&data.dev_inSrc, bitmap.image_size() ));
	HANDLE_ERROR(
		cudaMalloc( (void**)&data.dev_outSrc, bitmap.image_size() ));
	HANDLE_ERROR(
		cudaMalloc( (void**)&data.dev_constSrc, bitmap.image_size() ));

	// new additions for texture memory
	// instruct runtime that buffer we plan to use will be treated as 2-dim. texture, and 
	// bind 3 allocations to texture references declared earlier
	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
	HANDLE_ERROR( 
					cudaBindTexture2D(	NULL, texConstSrc,
										data.dev_constSrc,
										desc, DIM, DIM,
										sizeof(float) * DIM ));
										
	HANDLE_ERROR( 
					cudaBindTexture2D(	NULL, texIn,
										data.dev_inSrc,
										desc, DIM, DIM,
										sizeof(float) * DIM ));

	HANDLE_ERROR( 
					cudaBindTexture2D(	NULL, texOut,
										data.dev_outSrc,
										desc, DIM, DIM,
										sizeof(float) * DIM ));

	// initialize the constant data
	float *temp = (float*)malloc( bitmap.image_size() );
	for (int i=0; i<DIM*DIM; i++) {
		temp[i] = 0;
		int x = i % DIM;
		int y = i / DIM;
	// initial conditions
		if((x>300) && (x<600) && (y>310) && (y<601))
			temp[i] = MAX_TEMP;
	}
	// more initial conditions
	temp[DIM*100+100] = (MAX_TEMP + MIN_TEMP)/2;
	temp[DIM*700+100] = MIN_TEMP;
	temp[DIM*300+300] = MIN_TEMP;
	temp[DIM*200+700] = MIN_TEMP;
	for (int y=800; y<900; y++) {
		for (int x=400; x<500; x++) {
			temp[x+y*DIM] = MIN_TEMP;
		}
	}
	HANDLE_ERROR( 
		cudaMemcpy( data.dev_constSrc, temp, bitmap.image_size(), cudaMemcpyHostToDevice ));
		
	// initialize the input data	
	for (int y=800; y<DIM; y++) {
		for (int x=0; x<200; x++) {
			temp[x+y*DIM] = MAX_TEMP;
		}
	}
	HANDLE_ERROR(
		cudaMemcpy( data.dev_inSrc, temp, bitmap.image_size(), cudaMemcpyHostToDevice ));
	
	free( temp );
	
	bitmap.anim_and_exit( (void (*)(uchar4*, void*,int))anim_gpu, NULL );
}

