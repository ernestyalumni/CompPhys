/* cf. Jason Sanders, Edward Kandrot. CUDA by Example: An Introduction to General-Purpose GPU Programming */
/* 3.2.2 A Kernel Call */
#include <stdio.h>

__global__ void kernel(void) {
}

int main(void) {
	kernel<<<1,1>>>();
	printf("Hello, World!\n");
	return 0;
}
