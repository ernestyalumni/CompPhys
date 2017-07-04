# `CompPhys/moreCUDA/CUSOLVER/`
==============================
- Includes pedagogical (and simple) examples of using `cuSOLVER`

## CUDA Toolkit Documentation/API for `cuSOLVER`

I refer back to the CUDA Toolkit Documentation directly, and click through the pages to find what I need so much that I took notes here.

cf. [cuSolverDN: Dense LAPACK](http://docs.nvidia.com/cuda/cusolver/index.html#cuds-intro)

### [`cusolverDn<t>gesvd()`](http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-gesvd)

cf. [`cusolverDn<t>gesvd()`](http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-gesvd)

Note that the `<t>` means it's a wildcard, i.e. place holder, for what `<t>` could possibly be: in this case it can be either "S","D","C", or only "Z" (no quotation marks, only the single letter itself).  This wasn't obvious to be the first time.    
    
"S" - real-valued single precision
"D" - real-valued double precision
"C" - complex-valued single precision
"Z" - complex-valued double precision   

So this is documentation for *4* different functions,
- `cusolverDnSgesvd`   
- `cusolverDnDgesvd`
- `cusolverDnCgesvd`
- `cusolverDnZgesvd`   

*First*, these are the the helper functions below that can calculate the sizes needed for pre-allocated buffer.   


```   
cusolverStatus_t
cusolverDnDgesvd_bufferSize(
	cusolverDnHandle_t handle,
	int m,   
	int n,
	int *lwork);


cusolverStatus_t
cusolverDnSgesvd_bufferSize(
	cusolverDnHandle_t handle,
	int m,   
	int n,
	int *lwork);

cusolverStatus_t
cusolverDnDgesvd_bufferSize(
	cusolverDnHandle_t handle,
	int m,
	int n,
	int *lwork );   

cusolverStatus_t
cusolverDnCgesvd_bufferSize(
	cusolverDnHandle_t handle,
	int m,
	int n,
	int *lwork );   

cusolverStatus_t
cusolverDnZgesvd_bufferSize(
	cusolverDnHandle_t handle,
	int m,
	int n,
	int *lwork );   
```   
```   
cusolverStatus_t
cusolverDnDgesvd (
	cusolverDnHandle_t handle,
	signed char jobu,
	signed char jobvt,
	int m,   
	int n,
	double *A,
	int lda,
	double *S,
	double *U,
	int ldu,
	double *VT,
	int ldvt,
	double *work,  int lwork,
	double *rwork,
	int *devInfo);

```   

The C and Z data types are complex valued single and double precision, respectively.  
```   
cusolverStatus_t
cusolverDnCgesvd (
	cusolverDnHandle_t handle,   
	signed char jobu,   
	signed char jobvt,   
	int m,   
	int n,   
	cuComplex *A,   
	int lda,
	float *S,   
	cuComplex *U,
	int ldu,
	cuComplex *VT,
	int ldvt,
	cuComplex *work,
	int lwork,
	float *rwork,
	int *devInfo);   
   
cusolverStatus_t
cusolverDnZgesvd(
	cusolverDnHandle_t handle,
	signed char jobu,   
	signed char jobvt,
	int m,   
	int n,
	cuDoubleComplex *A,
	int lda,
	double *S,
	cuDoubleComplex *U,
	int ldu,
	cuDoubleComplex *VT,
	int ldvt,
	cuDoubleComplex *work,
	int lwork,
	double *rwork,
	int *devInfo);   
```   

This function computes the singular value decomposition (SVD) of a `mxn` matrix `A` and corresponding the left and/or right singular vectors.  The SVD is written

$$
	A = U * \Sigma * V^H   
$$   
i.e.

	A = U * S * V^H

where $\Sigma$ or S is a `mxn` matrix which is 0 except for its `min(m,n)` diagonal elements, `U` is a `mxm` unitary matrix, and `V` is a `nxn` unitary matrix.  

Remark 1: **`gesvd` only supports `m>=n`.  


**API of gesvd**   

| parameter        | Memory  | In/out | Meaning |
| ---------------- | :------------: | :---------------------- | :---------- |
| `jobu`  | host   | input  | specifies options for computing all or part of the matrix `U`     
  	    	     	      := `A`: all m columns of U are returned in array U    
			      := `S`: the first min(m,n) columns of U (the left singular vectors are returned in the array U;   
			      := `O`: the first min(m,n) columns of U (the left singular vectors) are overwritten on the array A;
		      	      := `N`: no columns of U (no left singular vectors) are computed.    |
| `m`     | host    | input  | number of rows of matrix `A` |   
| `n`     | host    | input  | number of columns of matrix `A` |
| `A`     | device  | in/out | <type> array of dim. `lda * n` with `lda` is not less than `max(1,m)`.  On exit, the contents of `A` are destroyed |  
| `lda`     | host   | input  | leading dim. of 2-dim. array used to store matrix A |
| `S`     | device | output | real array of dimension `min(m,n)`.  The singular values of A, sorted so that `S(i) >= S(i+1)`.  |
| `U`     | device | output | <type> array of dim. `ldu * m` with `ldu` is not less than `max(1,m)`.  `U` contains the `mxm` unitary matrix `U`.  |
| `ldu`   | host   | input  | leading dim. of 2-dim. array used to store matrix `U`. |
| `VT`    | device | output | <type> array of dim. `ldvt * n` with `ldvt` is not less than `max(1,n)`.  `VT` contains the `nxn` unitary matrix V**T.  |
| `ldvt`  | host | input | leading dim. of 2-dim. array used to store matrix `Vt`. |
| `work`  | device | in/out | working space, <type> array of size `lwork`     |
| `lwork` | host   | input  | size of `work`, returned by `gesvd_bufferSize`. |

#### `gesvd` My notes

```
cusolverStatus_t cusolverDnDgesvd(cusolverDnHandle_t handle,
					signed char jobu, signed char jobvt,
					int m,
					int n,
					double *A,
					int lda, // leading dim. of 2-dim. array used to store matrix A
					double *S,
					double *U,
					int ldu,
					double *VT,
					int ldvt,
					double *work,
					int lwork,
					double *rwork,
					int *devInfo);

```   
##### Summary of `cusolverDn<t>gesvd()`

- *Inputs*
  * `(m,n)`, `m>=n` ('gesvd', 'cusolver' requirement), `lda=m` (usually)
  * `A - mxn` Matrix, `m>=n` (`gesvd`, `cusolver` requirement)
- *Outputs*
  * `S - min(m,n)=n` real (float or double) array
  * `U - mxm` matrix, `ldu=m` (usually)
  * `VT - nxn` matrix, `ldvt=n` (usually)  


## Notes on compiling `cuSOLVER` scripts (with `nvcc`)

Currently (20170703), I use Fedora 23 Workstation (with a 980Ti), where I have manually, in root, made various symbolic links haphazardly, and on the Dell Inspiron 15 in. 7000 Gaming (1050; Dell, please sponsor me and my scientific and engineering endeavors!), Ubuntu 16.02 LTS, where I only do `apt-get` first, and maintains this OS with as little changes as possible (adding new software, no manually changes to `/usr/*`, etc.).  I will refer to Fedora first.  

[`SVD_vectors.cu`](https://github.com/ernestyalumni/CompPhys/blob/master/moreCUDA/CUSOLVER/SVD_vectors.cu) is the gold standard, being an exact copy of the example given in the [CUDA Toolkit documentation](http://docs.nvidia.com/cuda/cusolver/index.html#svd_examples).  

This worked (on Fedora 23):
```
nvcc -c SVD_vectors.cu   
g++ -fopenmp -o a.out SVD_vectors.o -L/usr/local/cuda/lib64 -lcudart -lcublas -lcusolver   
```   

When I did this, the following error(s) was obtained:
```
nvcc -c -I/usr/local/cuda/include SVD_vectors.cu   
/usr/local/cuda/include/surface_functions.h(134): error: expected a ";"
```   

But this also worked:   
```  
nvcc -std=c++11 -arch='sm_52' -lcublas -lcusolver SVD_vectors.cu -o SVD_vectors.exe   
```

This get tricky when implementing CUDA Unified Memory Management:  consider [`SVD_vectors_unified.cu`](https://github.com/ernestyalumni/CompPhys/blob/master/moreCUDA/CUSOLVER/SVD_vectors_unified.cu)

I found this to work, computing `U,S,VT` correctly, but not the error checks:
```
nvcc -std=c++11 -arch='sm_52' -lcudart -lcublas -lcusolver SVD_vectors_unified.cu -o SVD_vectors_unified.exe
```

- [`SVD_eg_unified.cu`](https://github.com/ernestyalumni/CompPhys/blob/master/moreCUDA/CUSOLVER/SVD_eg_unified.cu)

