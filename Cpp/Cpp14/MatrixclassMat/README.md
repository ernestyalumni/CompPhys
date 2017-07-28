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

Then, we want to input in the numerical values for the entries of this matrix as a 1-dimensional vector (`std::vector` in C++11), in *row-major ordering*.  What's that?  Assuming 0-based counting (i.e. in C++, Python, etc., we count starting from 0, 0,1,...), saw we have 6 entries in a 3x2 matrix.  Starting from a 1-dim. vector `a_vec` with the entries laid out as such:
```
index : 0        | 1        | 2        | 3        | 4        | 5        |
a_vec : a_vec[0] | a_vec[1] | a_vec[2] | a_vec[3] | a_vec[4] | a_vec[5] |  
      : 1        | 2        | 3        | 4        | 5        | 6        |  
```   

