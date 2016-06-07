/* cf. Jason Sanders, Edward Kandrot. CUDA by Example: An Introduction to General-Purpose GPU Programming */
/* 
** Chapter 4 Parallel Programming in CUDA C
** 4.2 CUDA Parallel Programming
** 4.2.1 Summing Vectors
*/
#include <stdio.h>

#define N  10

void add(int *a, int*b, int *c){
  int tid = 0;  // this is CPU zero, so we start at zero
  while (tid < N) {
    c[tid] = a[tid] + b[tid];
    tid += 1;   // we have one CPU, so we increment by one
  }
}
  

int main(void) {
  int a[N], b[N], c[N];

  // fill the arrays 'a' and 'b' on the CPU
  for (int i=0; i<N; i++) {
    a[i] = -i;
    b[i] = i * i;  
  }
  add(a,b,c);
  // display the results
  for (int i=0; i<N; i++) {
    printf( "%d + %d = %d\n", a[i], b[i], c[i] );
  }

  return 0;
}
  

