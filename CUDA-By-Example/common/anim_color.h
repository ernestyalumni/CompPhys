/* cf. Jason Sanders, Edward Kandrot. CUDA by Example: An Introduction to General-Purpose GPU Programming */
// I found this copyright notice off of jiekebo's repo:
/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.
 * Any use, reproduction, disclosure, or distribution of this software
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA)
 * associated with this source code for terms and conditions that govern
 * your use of this NVIDIA software.
 *
 */
#ifndef  __ANIM_COLOR_H__
#define __ANIM_COLOR_H__
 
template< typename T >
void swap( T& a, T& b) {
	T t = a;
	a = b;
	b = t;
}

// a place for common kernels - starts here

__device__ unsigned char value( float n1, float n2, int hue ){
	if (hue > 360)     hue -= 360;
	else if (hue < 0)  hue += 360;
	
	if (hue < 60)
		return (unsigned char)(255 * (n1 + (n2-n1)*hue/60));
	if (hue < 180)
		return (unsigned char)(255 * n2);
	if (hue < 240)
		return (unsigned char)(255 * (n1 + (n2-n1)*(240-hue)/60));
	return (unsigned char)(255 * n1);
}

__global__ void float_to_color( unsigned char *optr,
								const float *outSrc) {
	// map from threadIdx/BlockIdx to pixel position
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;
	
	float l = outSrc[offset];
	float s = l;
	int h = (180 + (int)(360.0f * outSrc[offset])) % 360;
	float m1, m2;
	
	if (l <= 0.5f)
		m2 = l * (l + s);
	else
		m2 = l + s - l * s;
	m1 = 2 * l - m2;
	
	optr[offset*4 + 0] = value( m1, m2, h+120 );
	optr[offset*4 + 1] = value( m1, m2, h );
	optr[offset*4 + 2] = value( m1, m2, h - 120 );
	optr[offset*4 + 3] = 255;
}

__global__ void float_to_color( uchar4 *optr, 
								const float *outSrc) {
	// map from threadIdx/BlockIdx to pixel position
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;
	
	float l = outSrc[offset];
	float s = l;
	int h = (180 + (int)(360.0f * outSrc[offset])) % 360;
	float m1, m2;
	
	if (l <= 0.5f)
		m2 = l * (1 + s);
	else
		m2 = l + s - l * s;
	m1 = 2 * l - m2;
	
	optr[offset].x = value( m1, m2, h+120 );
	optr[offset].y = value( m1, m2, h );
	optr[offset].z = value( m1, m2, h - 120 );
	optr[offset].w = 255;
}

#endif // __ANIM_COLOR_H__
