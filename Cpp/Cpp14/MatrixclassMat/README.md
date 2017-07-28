# `Mat` - C++11 class template for Matrix multiplication and matrix transpose

## Quick User's, Usage, guide

To use the class template `Mat` found in `./Mat/Mat.h` (`Mat.h` is the header file), you can do the following:
* copy and paste header file `Mat.h` to your desired directory for your code.  Then add an include (i.e. `#include`) line at the top of your code.  For example, if you put the directory `Mat` into the directory containing your code, do this:
```
#include './Mat/Mat.h`
```
- Compile your file (call it `main_Mat.cpp` for this example) with the following line:   
```
g++ -std=c++11 main_Mat.cpp
```
* copy and paste header file `Mat.h` into your "root" directories of "includes" if you have administrator privileges (usually it's `/usr/include/`; be sure to check also your bash profile, if you're on Mac OSX/Linux to see what else it includes).  Then add this line to the top of your code:
```
#include <Mat.h>
```
- Compile your file (call it `main_Mat.cpp` for this example) with the following line:
```
g++ -std=c++11 main_Mat.cpp -lMat
```

In my subjective opinion, I would only recommend the first way because it is unclear (as it's system setup dependent) what dependencies could be affected when changing files in root with an administrator account.

### Making Matrices with `Mat`

You will definitely need to know the *matrix size dimensions*, (M,P) or M x P or i.e the number of rows x number of columns, of the desired matrix.  For example, say you have `M=3` and `P=2`, i.e. the number of rows is 3 and number of columns is 2.

Then, we want to input in the numerical values for the entries of this matrix as a 1-dimensional vector (`std::vector` in C++11), in *row-major ordering*.  What's that?  Assuming 0-based counting (i.e. in C++, Python, etc., we count starting from 0, 0,1,...), saw we have 6 entries in a 2x3 matrix.  Starting from a 1-dim. vector `a_vec` with the entries laid out as such:
```
index : 0        | 1        | 2        | 3        | 4        | 5        |
a_vec : a_vec[0] | a_vec[1] | a_vec[2] | a_vec[3] | a_vec[4] | a_vec[5] |  
      : 1        | 2        | 3        | 4        | 5        | 6        |  
```   

When you do row-major ordering, it becomes the following matrix:
```
[ 1 2 3 ]
[ 4 5 6 ]
```
See how the entries in a single row are "continguous in the memory address" for C++.

Knowing what row-major ordering is, make a `std::vector<Type>` with `Type` being what type of numbers you want to compute (e.g. `float`, `int`, etc.) with the entries; for example
```
std::vector<float> entries {1,2,3,4,5,6};   
```   

Then here's the syntax for the `Mat` class template:
```
Mat<Type> (unsigned int first_dim, unsigned int second_dim,
	  	    std::vector<Type> & Entries)
```
with parameters
- `unsigned int first_dim` - first size dimension of the matrix
- `unsigned int second_dim` - second size dimension of the matrix
- `std::vector<Type> Entries` - matrix entries as a 1-dim. `std::vector`, assuming *row-major ordering*.

Useful methods for getting information about the matrix is as follows:
- `.get_first_dim()` - returns first size dimension of the matrix as an `unsigned int`
- `.get_second_dim()` - returns second size dimension of the matrix as an `unsigned int`
- `.print_all()` - pretty prints the entire matrix

### Matrix Multiplication

Matrix multiplication is easy! (as Object-Oriented Programming (OOP) should do)  Make 2 matrices and be sure their "inner size dimensions" are the same (I assume this is the case in the code because you should know some high school linear algebra before doing matrix multiplication).

Make 2 matrices.  For example, creating matrices `A` and `B`
```
std::vector<int> A_entries {1,2,3,4,5,6,7,8,9,10,11,12};
std::vector<int> B_entries {11,22,33,44,55,66};

Mat<int> A(4,3,A_entries);
Mat<int> B(3,2,B_entries);
```
Multiply them together like this:
```
A*B ;
```  
Indeed, do `.print_all()` on that:
```   
(A*B).print_all();
```
which results in
```
242 308
539 704
836 1100
1133 1496  
```

which can be confirmed by hand or in Python! See the file `Mat_playground.ipynb`.

### Transpose of a matrix `.T()`

Getting the transpose of a matrix is easy.  Given that previous matrix `A`, do this:
```
A.T();
```
That's it!

See the Developer's Manual, `Mat_dev_manual.pdf`, for an indepth look at the mathematics and a few notes for developers.


