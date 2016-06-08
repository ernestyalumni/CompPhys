/*
 * add-passb.cu
 */
/* cf. Jason Sanders, Edward Kandrot. CUDA by Example: An Introduction to General-Purpose GPU Programming */
/* 3.2.2 A Kernel Call */
#include <stdio.h>
#include "common/errors.h"

__global__ void add(int a, int b, int *c) {
	*c = a + b;
}

int main(void) {
	int c;
	int *dev_c;

	HANDLE_ERROR(
		     cudaMalloc((void**)&dev_c, sizeof(int))
		     );

	add<<<1,1>>>(2,7,dev_c);

	HANDLE_ERROR(
		     cudaMemcpy(&c, dev_c, sizeof(int),cudaMemcpyDeviceToHost)
		     );

	printf("2 + 7 =%d\n",c);
	cudaFree(dev_c);

	return 0;
}
