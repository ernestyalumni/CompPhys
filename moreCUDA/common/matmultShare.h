/* multShare.h
 * Matrix Multiplication using Shared Memory 
 * based on code from CUDA C Programming Guide
 * Ernest Yeung  ernestyalumni@gmail.com
 * 20160620
*/
typedef struct { 
	int Ni;
	int Nj;
	float* elements;
	int stride;
} Matrix;

__global__ void MatMulkernel(const Matrix, const Matrix, Matrix);
