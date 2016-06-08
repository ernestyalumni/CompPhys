/*
 * query.cu
 */
/* cf. Jason Sanders, Edward Kandrot. CUDA by Example: An Introduction to General-Purpose GPU Programming */
/* 3.3 Querying Devices 
** pp. 31 */
#include <stdio.h>
#include "common/errors.h"

int main(void) {
  cudaDeviceProp prop;

  int count;
  HANDLE_ERROR(
	       cudaGetDeviceCount( &count )
	       );

  for (int i = 0 ; i < count ; i++ ) {
    HANDLE_ERROR(
		 cudaGetDeviceProperties( &prop, i )
		 );
    printf( "Name: %s\n", prop.name );
    
  }
}
