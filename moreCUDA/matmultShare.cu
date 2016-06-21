/* matrixmultShare.h
 * Matrix Multiplication using Shared Memory 
 * based on code from CUDA C Programming Guide
 * Ernest Yeung  ernestyalumni@gmail.com
 * 20160620
*/
#include "./common/matmultShare.h"
#include "./common/errors.h" /* HANDLE_ERROR */
#include <stdlib.h> /* random */
#include <stdio.h> /* printf */

// Thread block size
#define BLOCK_SIZE 4

// Matrix multiplication - Host code
// Matrix dimensions assumed to be multiples of BLOCK_SIZE in matmultShare.h
void MatMul(const Matrix A, const Matrix B, Matrix C) {
	// Load A and B to device memory
	Matrix d_A;
	d_A.Ni = A.Ni;
	d_A.Nj = d_A.stride = A.Nj;
	
	size_t size = A.Ni*A.Nj*sizeof(float);
	
	HANDLE_ERROR( 
		cudaMalloc(&d_A.elements, size) );

	cudaMemcpy( d_A.elements, A.elements, size, cudaMemcpyHostToDevice);
	
	Matrix d_B;
	d_B.Ni = B.Ni;
	d_B.Nj = d_B.stride = B.Nj ;

	size = B.Ni*B.Nj*sizeof(float);
	HANDLE_ERROR(
		cudaMalloc(&d_B.elements, size ));
	
	cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);
	
	// Allocate C in device memory
	Matrix d_C;
	d_C.Nj = d_C.stride = C.Nj;
	d_C.Ni = C.Ni;
	size = C.Ni*C.Nj*sizeof(float);
	HANDLE_ERROR(
		cudaMalloc(&d_C.elements,size) );
	
	// Invoke kernel
	dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
	dim3 dimGrid(B.Nj / dimBlock.x, A.Ni / dimBlock.y );

	MatMulkernel<<<dimGrid,dimBlock>>>(d_A,d_B,d_C);
	HANDLE_ERROR( 
		cudaThreadSynchronize() );

	//  Read C from device memory
	HANDLE_ERROR(
		cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost) );

	// Free device memory
	cudaFree(d_A.elements);
	cudaFree(d_B.elements);
	cudaFree(d_C.elements);	
}

// Get a matrix elements

__device__ float GetElement( const Matrix A, int i, int j ){
	return A.elements[i*A.stride + j];
}

// Set a matrix element

__device__ void SetElement(Matrix A, int i, int j, float value) {
	A.elements[i*A.stride + j] = value;
}

// Get the BLOCK_SIZE*BLOCK_SIZE sub-matrix Asub of A that is 
// located col sub-matrices to the right and row sub-matrices down
// from the upper-left corner of A

__device__ Matrix GetSubMatrix(Matrix A, int i, int j) {
	Matrix Asub;
	Asub.Ni = BLOCK_SIZE;
	Asub.Nj = BLOCK_SIZE;
	Asub.stride = A.stride;
	Asub.elements = &A.elements[A.stride*BLOCK_SIZE*i+BLOCK_SIZE*j];
	return Asub;
}

// Matrix multiplication kernel called by MatMul()
__global__ void MatMulkernel(Matrix A, Matrix B, Matrix C) {
	// Block row and column
	int blockI = blockIdx.y;
	int blockJ = blockIdx.x;

	// Each thread block computes one sub-matrix Csub of C
	Matrix Csub = GetSubMatrix(C, blockI, blockJ);
	
	// Each thread computes 1 element of Csub
	// by accomulating results into Cvalue
	float Cvalue = 0.0;
	
	// Thread row and column within Csub
	int i = threadIdx.y;
	int j = threadIdx.x;
	
	// Loop over all the sub-matrices of A and B that are 
	// required to compute Csub
	// Multiply each pair of sub-matrices together
	// and accumulate the results
	for (int m = 0; m < (A.Nj / BLOCK_SIZE);++m) {
		// Get sub-matrix Asub of A
		Matrix Asub = GetSubMatrix(A, blockI, m);
		
		// Get sub-matrix Bsub of B
		Matrix Bsub = GetSubMatrix(B,m,blockJ);
		
		// Shared memory used to store Asub and Bsub respectively
		__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
		
		// Load Asub and Bsub from device memory to shared memory
		// Each thread loads one element of each sub-matrix
		As[i][j] = GetElement(Asub,i,j);
		Bs[i][j] = GetElement(Bsub,i,j);
		
		// Synchronize to make sure the sub-matrices are loaded
		// before starting the computation
		__syncthreads();
		
		// Multiply Asub and Bsub together
		for (int k = 0; k<BLOCK_SIZE; ++k)
			Cvalue += As[i][k]*Bs[k][j];
			
		// Synchronize to make sure that the preceding 
		// computation is done before loading 2 new
		// sub-matrices of A and B in the next iteration
		__syncthreads();
	}
	
	// Write Csub to device memory
	// Each thread writes 1 element
	SetElement(Csub,i,j,Cvalue);
}

/* display usage */
int help() {
	printf("Usage: matmultShare aNi aNj bNj \n Usage: matmultShare [-n1 -n2 -n3]\n");
	printf("\t -n1 -n2 -n3: are (3) numbers \n");
	return 1;
}

// Usage: matmultShare aNi aNj bNj e.g. matmultShare 5 4 3
int main(int argc, char* argv[]) {
	if (argc < 4) {
		return help();
	}
	
	Matrix A,B,C;
	int aNi, aNj, bNi, bNj;
	aNi = atoi(argv[1]); /* Ni of A */	
	aNj = atoi(argv[2]); /* Nj of A */
	bNi = aNj; /* Ni of B */
	bNj = atoi(argv[3]); /* Nj of B */
	
	A.Ni = aNi;
	A.Nj = aNj;
	A.elements = (float* )malloc(A.Ni*A.Nj*sizeof(float));
	
	B.Ni = bNi;
	B.Nj = bNj;
	B.elements = (float* )malloc(B.Ni*B.Nj*sizeof(float));
	
	C.Ni = A.Ni;
	C.Nj = B.Nj;
	C.elements = (float* )malloc(C.Ni*C.Nj*sizeof(float));
	
	for (int i = 0; i < A.Ni; i++ )
		for (int j = 0; j < A.Nj; j++)
			A.elements[i*A.Nj + j] = rand() % 5 + 1;
			
	for (int i = 0; i < B.Ni; i++ )
		for (int j = 0; j < B.Nj; j++)
			B.elements[i*B.Nj + j] = rand() % 4 + 1;
	
	MatMul(A,B,C);
	
	for (int i = 0 ; i < min(10, A.Ni); i++ ) {
		for (int j = 0; j < min(10, A.Nj); j++ ) 
			printf("%f ", A.elements[i*A.Nj + j ]);
		printf("\n");
	}
	
	for (int i = 0 ; i < min(10, B.Ni); i++ ) {
		for (int j = 0; j < min(10, B.Nj); j++ ) 
			printf("%f ", B.elements[i*B.Nj + j ]);
		printf("\n");
	}
	
	for (int i = 0 ; i < min(10, C.Ni); i++ ) {
		for (int j = 0; j < min(10, C.Nj); j++ ) 
			printf("%f ", C.elements[i*C.Nj + j ]);
		printf("\n");
	}
	printf("\n");

}
	





