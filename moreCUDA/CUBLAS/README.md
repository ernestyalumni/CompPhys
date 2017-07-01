# cuBLAS, CUBLAS

| codename        | Key code, keywords, code function, demonstrations of | Description             |
| --------------- | :-------------------------------------: | :---------------------- |
| `001isamax.cu` | `cublasIsamax`,`cublasIsamin`                     | find smallest index of element of an array with max/min magnitude |

## CUDA Toolkit Documentation/API for `cuBLAS`

cf. [`cuBLAS`](http://docs.nvidia.com/cuda/cublas/index.html#abstract)

I refer back to the CUDA Toolkit Documentation directly, and click through the pages to find what I need so much that I took notes here.

### [`cublas<t>dgmm()`](http://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-dgmm)  

```
cublasStatus_t cublasDdgmm(cublasHandle_t handle, cublasSideMode_t mode,
	       			int m, int n,
				const double	*A, int lda,
				const double 	*x, int incx,
				double 		*C,int ldc)

cublasStatus_t cublasCdgmm(cublasHandle_t handle, cublasSideMode_t mode,
	       			int m, int n,
				const cuComplex	*A, int lda,
				const cuComplex 	*x, int incx,
				cuComplex 		*C,int ldc)

cublasStatus_t cublasZdgmm(cublasHandle_t handle, cublasSideMode_t mode,
	       			int m, int n,
				const cuDoubleComplex *A, int lda,
				const cuDoubleComplex *x, int incx,
				cuDoubleComplex *C, int ldc)
   
```   

This function performs the matrix-matrix multiplication
$$   
C = \begin{cases}
  A \times \text{diag}(X)  & \text{ if mode } == \verb|CUBLAS_SIDE_RIGHT| \\
  \text{diag}(X) \times A & \text{ if mode } == \verb|CUBLAS_SIDE_LEFT|
  \end{cases}
$$
i.e.   
if mode `== CUBLAS_SIDE_RIGHT`,
C = A x diag(X)
if mode `== CUBLAS_SIDE_LEFT`,
C = diag(X) x A

i.e. **`cublas<t>dgmm()`** does matrix x *diagonal matrix* multiplication

| parameter | Memory  | In/out | Meaning |
| --------- | :------------: | :---------------------- | :---------- |
| x         | device  | input  | 1-dim. <type> array of size |inc| x m if `mode == CUBLAS_SIDE_LEFT` and |inc| x n if `mode == CUBLAS_SIDE_RIGHT` |
| incx      |         | input  | stride of 1-dim. array `x` |






