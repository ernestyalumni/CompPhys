# Cpp - C++ examples for Computational Physics and Computational Fluid Dynamics (CFD)

## Abstract

Included in this repository (repo) are code examples straight from M. Hjorth-Jensen's **Computational Physics**, University of Oslo (2015) [mhjensen](http://www.mn.uio.no/fysikk/english/people/aca/mhjensen/).  

Hjorth-Jensen's material motivated me to (try to) dive deeper into C++ and so I picked up Bjarne Stroustrup's **A Tour of C++** (Addison-Wesley Professional, 2013).  This book is based on **C++11** and I suspect that a lot of computational physics code was written *before* this 2011 (keep in mind that I still need to find a good book for C++14).  I include code for Stroustrup's **A Tour of C++** in the `./tour` subdirectory.  

You're (as I was) reading Hjorth-Jensen's excellent lecture notes (latest I found is from 2015) on Computational Physics (with an emphasis on C++).  You want to run the C++ code *simultaneously* as you are reading, as I found that I learn *much* faster watching how the code works than a book explanation - along with playing around with the code "in real-time" (it makes learning fun!).  But you don't know which code follows along which part of the lectures.  So I'll try to make a listing (a dictionary) or index of code to lecture, below.  

P.S. There are some non-trivial typos both in the code and lecture that made the material confusing to understand (as in, did he really mean that?) and some things weren't obvious to me when I read it, and so I try to make sense of it in my notes in [CompPhys.pdf](https://github.com/ernestyalumni/CompPhys/blob/master/LaTeXandpdfs/CompPhys.pdf) and have my own code type-ups here in this repository, even though there's the [CompPhys repository](https://github.com/CompPhysics).

## Listing of which program or script corresponds to which section, chapter, part for Hjorth-Jensen's material

See Hjorth-Jensen's lectures for 2015.  

| codename        | directory      | Chapter | Section | page (pp) | Description            |
| --------------- | -------------- | :-----: | ------- | --------- | ---------------------- |
| argcargv.cpp    | ./             |         |         |           | This is my own pedagogical example of using argc, argv in C++; it's also a good example of looping through an array as a pointer |
| program1.cpp    | ./progs/ch02/  | 2       | 2.1.1   | 10        | Scientific Hello World!|
| program7.cpp    | ./progs/ch02/  | 2       | 2.5.2   | 34        | Pointers and arrays in C++ |
| program7b.cpp   | ./progs/ch02/  | 2       | 2.5.2   | 34        | Pointers and arrays in C++ (further experimentation by me) |
| program1.cpp    | ./progs/ch03/  | 3       | 3.1.1.1 | 51        | calculates second derivative of exp(x); take note of C style file IO |
| program3.cpp    | ./progs/ch03/  | 3       | 3.1.1.4 | 54        | calculates second derivative of exp(x); take note of C++ style file IO |
| usecomplex.cpp  | ./progs/ch03/  | 3       | 3.3.1   | 68        | Using standard template library (STL) for <complex>, complex<double>  |
| usecomplexb.cpp | ./progs/ch03/  | 3       | 3.3.1   | 68        | More usage, such as x.real(), x.imag(), and exp(z) by me |
| complex.h       | ./progs/ch03/  | 3       | 3.3.1   | 67        | header file for Complex class |
| complex.cpp     | ./progs/ch03/  | 3       | 3.3.1   | 68        | contains Complex class |
| customCC.cpp    | ./progs/ch03/  | 3       | 3.3.1   | 68        | contains `main()` to demonstrate Complex class |
| Integrate.ipynb | ./             | 5       | 5.1     | 109-112   | jupyter (.ipynb) notebook to symbolic compute, using sympy, the derivation of trapezoid rule |
| program1.cpp    | ./progs/ch05/  | 5       | 5.3.6   | 127       | example using trapezoid rule, Simpson's rule, Gauss-Legendre method|
| integrate1d.h   | ./progs/ch05/  | 5       | 5.3.6   | 127       | header file for 1-dimensional numerical integration routines, only |
| integrate1d.cpp | ./progs/ch05/  | 5       | 5.3.6   | 127       | function definitions for 1-dimensional numerical integration routines, only |

## Listing of which program or script corresponds to which section, chapter, part for Stroustrup's **A Tour of C++** (2013)

| codename        | directory      | Chapter | Section | page (pp) | Description            |
| --------------- | -------------- | :-----: | ------- | --------- | ---------------------- |

### File I/O in C and C++

cf. `program1.cpp` in `./progs/ch03` subdirectory; pp. 51 Subsection 3.1.1.1 Initializations and main program, Ch. 3 Numerical differentiation and interpolation of Hjorth-Jensen (2015); [CompPhysics/ComputationalPhysicsMSU/doc/Programs/LecturePrograms/programs/Classes/cpp/
](https://github.com/CompPhysics/ComputationalPhysicsMSU/blob/master/doc/Programs/LecturePrograms/programs/Classes/cpp/program1.cpp)

Take a look at `program1.cpp`.  It appears that the "C"-style file IO is first considered.  Reading this wikipedia article on [C file input/output](https://en.wikipedia.org/wiki/C_file_input/output) helped me understand what was used here.

`FILE` - known as a file *handle*, abstract reference to a resource, this is an "opaque type" - points to some unspecified type - containing information about a file or text stream, and we can do IO (input output) operations on.

So `FILE *output_file;` declares a pointer `output_file` to a `FILE`. 

`stdio.h` contains `fopen`, `fprintf`, `fclose`:
- `fopen` - opens a file
- `fprintf` - prints formatted byte/wchar_t output to stdout, file stream, or buffer; syntax:

```
int fprintf_s(FILE *restrict stream, const char *restrict format, ...);
```
cf. http://en.cppreference.com/w/c/io/fprintf

- `fclose` - closes a file

cf. `program3.cpp` in `./progs/ch03` subdirectory, pp. 54-55, Subsection 3.1.1.4 The output function of Hjorth-Jensen (2015), [CompPhysics/ComputationalPhysicsMSU/doc/Programs/LecturePrograms/programs/Classes/cpp/
](https://github.com/CompPhysics/ComputationalPhysicsMSU/blob/master/doc/Programs/LecturePrograms/programs/Classes/cpp/program3.cpp)

#### `<fstream>` header

cf. http://www.cplusplus.com/reference/fstream/

`<fstream>` in `#include <fstream>` is a header providing file stream classes.  So apparently, for narrow characters [`char`], there is `fstream` (input/output file stream class) and `ofstream` (output file stream).

In `program3.cpp`, `ofile` is an "instance" of class `ofstream`:

```
ofstream ofile;
```
Here are examples of using this class `ofstream` (in this particular situation, its "instance" is called `ofile`):

```
ofile.open(outfilename);
ofile.close();
```
and then the actual "outputting":

```
ofile << setw(15) << setprecision(8) << log10(h_step[i]);
ofile << setw(15) << setprecision(8) << epsilon << endl;
```

`setw` is a class belonging to `<iomanip>`.  It stands for **Set field width** and sets the *field width* to be used on output operations (cf. http://www.cplusplus.com/reference/iomanip/setw/ ).  

### Classes (C++)

See `complex.h, complex.cpp, customCC.cpp` in `./progs/ch03` subdirectory for useful (pedagogical) examples.

## header files

Define header file (e.g. complex.h) which contains declarations of the class.
The header file contains
- class declaration (data and functions)
- declaration of stand-alone functions, and all inlined functions,

To understand

```
#ifndef HEADERFILE_H  
#define HEADERFILE_H  
```

See (http://stackoverflow.com/questions/1653958/why-are-ifndef-and-define-used-in-c-header-files) and (https://en.wikipedia.org/wiki/Include_guard).

### C++ Operator Overloading in expression; lvalues vs. rvalues

cf. [C++ Operator Overloading in expression](http://stackoverflow.com/questions/6377786/c-operator-overloading-in-expression)

Take a look at this link: [C++ Operator Overloading in expression](http://stackoverflow.com/questions/6377786/c-operator-overloading-in-expression).  This point isn't emphasized enough, as in Hjorth-Jensen (2015).  This makes doing something like  

```
d = a*c + d/b
```

work the way we expect.  Kudos to user [fredoverflow](http://stackoverflow.com/users/252000/fredoverflow) for his answer:

``The expression `(e_x*u_c)` is a rvalue, and references to non-const won't bind to rvalues.

Also, member functions should be marked `const` as well.''  

#### What are lvalues and rvalues in C and C++?

[C++ Rvalue References Explained](http://thbecker.net/articles/rvalue_references/section_01.html)

Original definition of *lvalues* and *rvalues* from *C*:  
*lvalue* - expression `e` that may appear on the left or on the right hand side of an assignment  
*rvalue* - expression that can only appear on right hand side of assignment =.

Examples:

```
  int a = 42;
  int b = 43;

  // a and b are both l-values
  a = b; // ok
  b = a; // ok
  a = a * b; // ok

  // a * b is an rvalue:
  int c = a * b; // ok, rvalue on right hand side of assignment
  a * b = 42; // error, rvalue on left hand side of assignment
  
```

In *C++*, this is still useful as a first, intuitive approach, but   
*lvalue* - expression that refers to a memory location and allows us to take the address of that memory location via the & operator.  
*rvalue* - expression that's not a lvalue

So & reference *functor* can't act on rvalue's.

### Results of Numerical Integration

See the lecture notes (2015) of Hjorth-Jensen; pp. 128, Ch. 5 Numerical Integration, Tables 5.2-5.3.  I wanted to reproduce those results.

My thought is that making a "smaller" file containing only the 1-dimensional (numerical) integration routines would be more useful (pedagogically).  Hence, including `integrate1d.h` header file in the working directory, you'd compile

```
g++ program1.cpp integrate1d.cpp
```

i.e. compile `program1.cpp` with `integrate1d.cpp` and then one can integrate your desired integrand, by including those "integrate1d" files.

My results are as follows:

For $\int_1^{100} \exp{(-x)/x \, dx$,

| N        | Trapezoid     | Simpson  | Rectangular | Gauss-Legendre |
| -------- | ------------- | :------: | ----------  | -------------- |
| 10       | 1.82102       | 0.607023 | 0.00433585  | 0.146045       |
| 20       | 0.912678      | 0.306397 | 0.0442333   | 0.217809       |
| 40       | 0.478456      | 0.181965 | 0.123049    | 0.219383       |
| 100      | 0.273724      | 0.170579 | 0.194204    | 0.219384       |
| 1000     | 0.219984      | 0.213317 | 0.219084    | 0.219384       |

For $\int_0^3 1/(2+x^2) \, dx$,

| N        | Trapezoid     | Simpson  | Rectangular | Gauss-Legendre |
| -------- | ------------- | :------: | ----------  | -------------- |
| 10       | 0.798861      | 0.769685 | 0.824581    | 0.799233       |
| 20       | 0.79914       | 0.78446  | 0.812373    | 0.799233       |
| 40       | 0.799209      | 0.791846 | 0.805925    | 0.799233       |
| 100      | 0.799229      | 0.796278 | 0.80194     | 0.799233       |
| 1000     | 0.799233      | 0.798937 | 0.799505    | 0.799233       |









