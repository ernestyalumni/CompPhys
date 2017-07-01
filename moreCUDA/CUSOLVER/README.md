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

```   
cusolverStatus_t
cusolverDnSgesvd_bufferSize(
	cusolverDnHandle_t handle,
	int m,   
	int n,
	int *lwork);



cusolverDnCgesvd_bufferSize(
	cusolverDnHandle_t handle,
	int m,
	int n,
	int *lwork );   

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
	cuComplex *A,
	int lda,
	double *S,
	cuComplex *U,
	int ldu,
	cuComplex *VT,
	int ldvt,
	cuComplex *work,
	int lwork,
	float *rwork,
	int *devInfo);

```

**API of gesvd**
| parameter        | Memory  | In/out | Meaning |
| --------------- | :------------: | :---------------------- | :---------- |
| `S` | `device`  | `output`   | real array of dimension `min(m,n)`.  The singular values of A, sorted so that `S(i) >= S(i+1)`.  |   
