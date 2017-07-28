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

