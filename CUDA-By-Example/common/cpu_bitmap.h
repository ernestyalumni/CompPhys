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
 
#ifndef __CPU_BITMAP_H__
#define __CPU_BITMAP_H__

#include <GL/glut.h>
#include <GL/glext.h>
#include <GL/glx.h>

struct CPUBitmap {
	unsigned char *pixels;
	int           x,y;
	void          *dataBlock;
	void (*bitmapExit)(void *);
	
	CPUBitmap( int width, int height, void *d = NULL ) {
		pixels = new unsigned char[width*height *4];
		x = width;
		y = height;
		dataBlock = d;
	}
	
	~CPUBitmap() {
		delete [] pixels;
	}
	
	unsigned char* get_ptr( void ) const { return pixels; }
	long image_size( void ) const { return x * y * 4; }
	
	void display_and_exit( void(*e)(void*) = NULL ) {
		CPUBitmap** bitmap = get_bitmap_ptr();
		*bitmap = this;
		bitmapExit = e;
		
		int c = 1;
		char * dummy = '\0';
		glutInit( &c, &dummy);
		glutInitDisplayMode( GLUT_SINGLE | GLUT_RGBA );
		glutInitWindowSize( x, y);
		glutCreateWindow( "bitmap" );
		glutKeyboardFunc( Key );
		glutDisplayFunc( Draw );
		glutMainLoop();
	}
	
	// static method used for glut callbacks
	static CPUBitmap** get_bitmap_ptr( void ) {
		static CPUBitmap    *gBitmap;
		return &gBitmap;
	}
	
	// static method used for glut callbacks
	static void Key(unsigned char key, int x, int y) {
		switch (key) {
			case 27:
			CPUBitmap* bitmap = *(get_bitmap_ptr());
			if (bitmap->dataBlock != NULL && bitmap->bitmapExit != NULL)
				bitmap->bitmapExit( bitmap->dataBlock );
			exit(0);
		}
	}
	
	// static method used for glut callbacks
	static void Draw( void ) {
		CPUBitmap* bitmap = *(get_bitmap_ptr());
		glClearColor( 0.0, 0.0, 0.0, 1.0 );
		glClear( GL_COLOR_BUFFER_BIT );
		glDrawPixels( bitmap->x, bitmap->y, GL_RGBA, GL_UNSIGNED_BYTE, bitmap->pixels);
		glFlush();
	}
};

#endif // __CPU_BITMAP_H__
		
