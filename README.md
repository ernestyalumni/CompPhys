# CompPhys
CompPhys - a Computational Physics repository.

Since being good at Computational Physics means being good at C/C++, and also C++11/C++14, which really feels like a new (and more modern and sensible language than before), and then CUDA C/C++ (which feels like new frontier), subdirectories in here has *lots* of (pedagogical) examples.  

## Contents

- Cpp
  * *Initially*, `Cpp/` directory has material based off of M. Hjorth-Jensen's **Computational Physics** [2015 lectures](https://github.com/CompPhysics/ComputationalPhysicsMSU/blob/master/doc/Lectures/lectures2015.pdf): what I did was to type up the (pedagogical) programs in the lectures, (try to) organize them so that you know which program corresponds to what part of the lectures (so that while you're reading the lectures, you can run the programs simultaneously), and compile and make sure they run (as I wanted them to run, myself).
  * Now, `Cpp/` includes more material, more or less related to C++: code snippets, pedagogical programs, programs and code exclusively built on the **C++11/C++14** standard, which is really a new language, and packages written entirely in C++ that I need practice with, such as *OpenCV* (Open Computer Vision).  
  * [C++ **Design Patterns**](https://github.com/ernestyalumni/CompPhys/tree/master/Cpp/DesignPatterns)  
  
- CUDA-By-Example
	* `CUDA-By-Example/` directory has material based on Sanders and Kandrot's **CUDA By Example**.  Most of the code examples from the book are there, and then some.  I truly believe this is the most comprehensive code repository (subdirectory) for codes out of that book that's out there.  
	* `CUDA-By-Example/common` : you should also note that the subdirectory `CUDA-By-Example/common` has useful header files and I try my best to explain thoroughly what the stuff in the header files mean: especially note `gpu_anim.h` because you can use that header file to do OpenGL and CUDA C bitmap animations that's **entirely rendered on the GPU.**  

- moreCUDA
	* `moreCUDA` directory has more CUDA C/C++ examples - I'm finding that as the CUDA Toolbox (version 7.5 at the time of writing this) adds more and more features, and with C++11/C++14, stuff from 2012, 2013 and before (it's 2016 at this time) isn't up to date, or coded to take advantage of C++11/C++14.  I'll try to alleviate this by showing examples in the `moreCUDA` directory there.  

- course_notes
  * **go here** for the latest implementations/examples of CUDA C/C++ - I continue to implement the latest features and ideas for CUDA C/C++; CUDA is moving so rapidly, much code from before, only 3-4 years ago, is becoming obsolete, or at least, not taking advantage of new GPU architecture.   I do this by going over other courses online.  They include
    - [HPC - Algorithms and Applications - Winter 16](https://www5.in.tum.de/wiki/index.php/HPC_-_Algorithms_and_Applications_-_Winter_16)

## C Resources

Coursera's Heterogeneous Parallel Programming (taught by Wen-mei W. Hwu) suggested these links to brush up on [C programming](https://class.coursera.org/hetero-004/wiki/Introduction_to_C):
-    http://www.cprogramming.com/tutorial/c-tutorial.html
-    http://www.physics.drexel.edu/courses/Comp_Phys/General/C_basics/
-    http://gd.tuwien.ac.at/languages/c/cref-mleslie/CONTRIB/SAWTELL/intro.html
-    http://en.wikibooks.org/wiki/C_Programming
-    http://www.cprogramming.com/
-    https://www.scaler.com/topics/c/

I also found this pdf from [Fitzpatrick, "Computational Physics"](http://farside.ph.utexas.edu/teaching/329/329.pdf).  It has fully working code.  So I will try to implement some of the code.  

### Pointers (for Computational Physics)

## GNU Scientific Library; GNU GSL 

I am using [GNU Scientific Library](http://www.gnu.org/software/gsl/manual/gsl-ref.pdf) (GNU GSL) because there are some serious concerns with [Numerical Reciples](http://www.lysator.liu.se/c/num-recipes-in-c.html): cf. https://www.reddit.com/r/Physics/comments/s9p16/the_numerical_recipes_license_is_the_riaa_of_the/ 

The rationale for GNU GSL is [clear and fairly straightforward](https://www.gnu.org/software/gsl/design/gsl-design.html).

### Compiling your C code using GSL, and making the executable to run in GSL

The [GNU GSL manual](http://www.gnu.org/software/gsl/manual/gsl-ref.pdf) is relatively straightforward and clear about the hoops you'll have to jump through to compile your C code that uses GSL, and making the executable so you can get a result.  For instance, on a Mac OS X, I typed up the example program on pp. 4, Section 2.1 (it's saved as `gsl_bessel_example.c` in this repository) and I ran these 2 commands:  
```
gcc -Wall -I/usr/local/include -c gsl_bessel_example.c  
gcc -L/usr/local/lib example.o -lgsl -lgslcblas -lm  
```    

The manual explains thoroughly and understandably what the flags in the command mean.  Note that when I removed the `-c` flag in the first command, hoping to not only create the `.o` object file, but to automatically make the executable, I was greeted with an error (!!!).  

```   
ld: symbol(s) not found for architecture x86_64  
clang: error: linker command failed with exit code 1 (use -v to see invocation)  
```    

### Let's start from the beginning...Installing GNU GSL onto Fedora 23 Linux
This install will probably work for any UNIX-type computer you have, with `gcc` and `g++` compilers ready to go.

The [GNU GSL website](http://www.gnu.org/software/gsl/) only mentioned the `INSTALL` and `README` file for the instructions.  Hilariously, the `README` file refers back to the `INSTALL` file for installation instructions.

I followed the instructions on the `README` file for GNU GSL:
```
./configure
make
make install
```

except for `make`, I did `make -j12` (to take advantage of multicores) and
I had to do `sudo make install` because only then, it had permission to change `/usr/local/include`, adding a new gsl subdirectory (it wasn't there before) with headers (header files) for gsl; also `/usr/bin/install` was involved, including `/usr/local/share` in the install.  

### (More) advice on compiling files with GNU GSL libraries

This command worked in compiling C files that uses the matrices library associated with GSL:
```
$ gcc -L/usr/local/lib matrices.c -lgsl -lgslcblas
```
I believe the `-L` flag helps to include the `/usr/local/lib` directory for the libraries while compiling with gcc and `-lgsl` and `-lgslcblas` helps to include the gsl headers and CBLAS headers.

Otherwise, when I didn't, I obtained these errors:
```
$ gcc matrices.c
/tmp/ccj9U9ED.o: In function `main':
matrices.c:(.text+0x13): undefined reference to `gsl_matrix_alloc'
matrices.c:(.text+0x6b): undefined reference to `gsl_matrix_set'
matrices.c:(.text+0xac): undefined reference to `gsl_matrix_get'
matrices.c:(.text+0xe3): undefined reference to `gsl_matrix_free'
collect2: error: ld returned 1 exit status
```

So after running `gcc -L/usr/local/lib matrices.c -lgsl -lgslcblas`, I thought I had a working executable `./a.out` but running it gave me this:
```
$ ./a.out
./a.out: error while loading shared libraries: libgsl.so.19: cannot open shared object file: No such file or directory
```

The [Shared Libraries](https://www.gnu.org/software/gsl/manual/html_node/Shared-Libraries.html) webpage gave me the answer, which was "To avoid this error, either modify the system dynamic linker configuration5 or define the shell variable `LD_LIBRARY_PATH` to include the directory where the library is installed."  Which I hadn't done yet.

This page, [Setting PATH and LD_LIBRARY_PATH for the bash shell](http://taopm.sourceforge.net/docs/online_userman/UserManual_13.html) gave possible solutions; the temporary solution I did was

```
$ LD_LIBRARY_PATH=/usr/local/lib
$ export LD_LIBRARY_PATH
$ ./a.out
```
and then the matrices.c script executed as desired.

### Setting the bash profile so you don't have to do the temporary solution above

I tried to look at these links:
- [2.3 Shared Libraries](https://www.gnu.org/software/gsl/manual/html_node/Shared-Libraries.html)
- [Setting PATH and LD_LIBRARY_PATH for the bash shell](http://taopm.sourceforge.net/docs/online_userman/UserManual_13.html)
- [How to set the environmental variable LD_LIBRARY_PATH in linux](http://stackoverflow.com/questions/13428910/how-to-set-the-environmental-variable-ld-library-path-in-linux)

The commands the links suggested, respectively are:
```
$ LD_LIBRARY_PATH=/usr/local/lib
$ export LD_LIBRARY_PATH
$ ./example

$ gcc -static example.o -lgsl -lgslcblas -lm
```
"To avoid this error, either modify the system dynamic linker configuration5 or define the shell variable `LD_LIBRARY_PATH` to include the directory where the library is installed."


```
echo $LD_LIBRARY_PATH

LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
export LD_LIBRARY_PATH
```

Ultimately, I did this
```
export LD_LIBRARY_PATH="/usr/local/lib:$LD_LIBRARY_PATH"
```
According to this page: [including usr/local/lib directory](http://www.linuxquestions.org/questions/linux-software-2/including-usr-local-lib-directory-272610/)

On that note, here are some commands I keep on using over and over on Fedora 23 Workstation Linux:


for THEANO, tensorflow:   
```   
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
echo $LD_LIBRARY_PATH
THEANO_FLAGS='mode=FAST_RUN,device=gpu,floatX=float32' python gpu_test.py   
```   
for CuDNN:    

```   
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
THEANO_FLAGS='mode=FAST_RUN,device=gpu,floatX=float32' jupyter notebook    
```

and in general:
```   
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib64

echo _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEED1Ev | c++filt

```    

## Getting started with `tmux` - simultaneous multiple terminal sessions

