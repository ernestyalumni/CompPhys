# Cpp - C++ examples for Computational Physics and Computational Fluid Dynamics (CFD)

## Abstract

Included in this repository (repo) are code examples straight from M. Hjorth-Jensen's **Computational Physics**, University of Oslo (2015) [mhjensen](http://www.mn.uio.no/fysikk/english/people/aca/mhjensen/).  

You're (as I was) reading Hjorth-Jensen's excellent lecture notes (latest I found is from 2015) on Computational Physics (with an emphasis on C++).  You want to run the C++ code *simultaneously* as you are reading, as I found that I learn *much* faster watching how the code works than a book explanation - along with playing around with the code "in real-time" (it makes learning fun!).  But you don't know which code follows along which part of the lectures.  So I'll try to make a listing (a dictionary) or index of code to lecture, below.  

P.S. There are some non-trivial typos both in the code and lecture that made the material confusing to understand (as in, did he really mean that?) and some things weren't obvious to me when I read it, and so I try to make sense of it in my notes in [CompPhys.pdf](https://github.com/ernestyalumni/CompPhys/blob/master/LaTeXandpdfs/CompPhys.pdf) and have my own code type-ups here in this repository, even though there's the [CompPhys repository](https://github.com/CompPhysics).

## Listing of which program or script corresponds to which section, chapter, part for Hjorth-Jensen's material

See Hjorth-Jensen's lectures for 2015.  

| codename        | directory      | Chapter | Section | page (pp) | Description            |
| --------------- | -------------- | :-----: | ------- | --------- | ---------------------- |
| program1.cpp    | ./progs/ch02/  | 2       | 2.1.1   | 10        | Scientific Hello World!|
| program7.cpp    | ./progs/ch02/  | 2       | 2.5.2   | 34        | Pointers and arrays in C++ |
| program7b.cpp   | ./progs/ch02/  | 2       | 2.5.2   | 34        | Pointers and arrays in C++ (further experimentation by me) |
| usecomplex.cpp  | ./progs/ch03/  | 3       | 3.3.1   | 68        | Using standard template library (STL) for <complex>, complex<double>  |
| usecomplexb.cpp | ./progs/ch03/  | 3       | 3.3.1   | 68        | More usage, such as x.real(), x.imag(), and exp(z) by me |
| complex.h       | ./progs/ch03/  | 3       | 3.3.1   | 67        | header file for Complex class |
| complex.cpp     | ./progs/ch03/  | 3       | 3.3.1   | 68        | contains Complex class |
| customCC.cpp    | ./progs/ch03/  | 3       | 3.3.1   | 68        | contains `main()` to demonstrate Complex class |


### Classes (C++)

See `complex.h, complex.cpp, customCC.cpp` in `./progs/ch03` subdirectory for useful (pedagogical) examples.

## header files

Define header file (e.g. complex.h) which contains declarations of the class.
The header file contains
- class declaration (data and functions)
- declaration of stand-alone functions, and all inlined functions,

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
