# Cpp14 - C++ code snippets and examples for C++11/C++14 standard

## Abstract

> C++ feels like a new language. That is, I can express my ideas more clearly, more simply, and more directly in C++11 than I could in C++98.  Furthermore, the resulting programs are better checked by the compiler and run faster. -*Bjarne Stroustrup*

This directory contains C++ code snippets and examples formulated entirely in the C++11/C++14 standard.  I sought to "borax" away any backwards-compatibility or previous code practices and to adopt anew best coding practices that are apt to this latest C++11/C++14 standard.  


## Table of Contents  

| codename         | directory      | external link(s) | Description             |
| ---------------- | -------------- | ---------------- | :----------------------: |
| `timingcode.cpp` | `./`           | [C++11 timing code performance](https://solarianprogrammer.com/2012/10/14/cpp-11-timing-code-performance/) | C++11 timing code performance |
| `vectors.cpp`    | `./`           | | Everything about vectors for C++11/14, cf. Lippman, Lajoie, Moo, **C++ Primer** 5th ed., *3.3. Library vector Type* |
| `vectors_binarysearch.cpp`    | `./`           | | classic algorithm, binary search, using vectors via C++11/14, cf. Lippman, Lajoie, Moo, **C++ Primer** 5th ed., *3.4. Introducing Iterators*, 3.4.2 |
| `functors.cpp`   | `./`           | [C++ Tutorial - Functors(Function Objects) - 2016 by K Hong?](http://www.bogotobogo.com/cplusplus/functors.php) | `functors, algorithm, operator()(), bind2nd` |
| `main.cu`	   | `../../moreCUDA/thruststuff/ranges` | [Separate C++ Template Headers (`*.h`) and Implementation files (`*.cpp`)](http://blog.ethanlim.net/2014/07/separate-c-template-headers-h-and.html) | other than examples of ranges, this is an example of separating C++ class templates to the header file |
| `bitwiserightshift.cpp` | `./` | [Udacity: Serial Implementation of Reduce](https://classroom.udacity.com/courses/cs344/lessons/86719951/concepts/876789040923#) | Demonstrates bitwise right shift and bitwise right shift assignment operators; Used in reduce.cu of Serial Implementation of Reduce |
| `unique_ptr.cpp ` | `./` |  [cppreference.com `std::unique_ptr`](http://en.cppreference.com/w/cpp/memory/unique_ptr) | `std::unique_ptr`, `unique_ptr`, `make_unique` |  
| `staticvars.cpp` | `./` | [What does “static” mean?](https://stackoverflow.com/questions/572547/what-does-static-mean) | `static` variables |  
| `FileIObin.cpp` | `./` | [Writing a binary file in C++ very fast](https://stackoverflow.com/questions/11563963/writing-a-binary-file-in-c-very-fast) | `std::iota` (fills the range `[first, last)` with sequentially increasing values, starting with `value` |  

### Basic ways to make or construct unique pointers to arrays:  

```  
auto uptr11 = std::unique_ptr<float[]>{ new float[42]};     // since C++11
``` 
```
auto uptr14 = std::make_unique<float[]>(42);                // since C++14
```  
And then access the entries this way:
```  
for (int i =0; i<42 ; i++) {
    std::cout << uptr11[i] << " " << uptr14[i] << " ";
    uptr11[i] = i*11.f;
    uptr14[i] = i*140.f;    
    std::cout << " " << i << " : " << uptr11[i] << " " << uptr14[i]  << std::endl;
}   
```  

## (even more) File I/O  

Please see [`CompPhys/Cpp/README.md`](https://github.com/ernestyalumni/CompPhys/blob/master/Cpp/README.md) for my original implementations of File I/O, especially the pure C++ implementations of File I/O for `.csv` files.  I will try to summarize and make concise those exact same contents (duplicate) here as well.  

## Glossary/Dictionary/Quick Look Up  

`override` - override specifier (>C++11) - specifies that virtual function overrides another virtual function, e.g.  
```  
struct A
{
    virtual void foo();
    void bar();
};

struct B : A
{
    void foo() const override;  // Error: B::foo does not override A::foo
                                // (signature mismatch)
    void foo() override;        // OK: B::foo overrides A::foo
    void bar() override;        // Error: A::bar is not virtual 
};
```  

`decltype` - `decltype` specifier (since C++11) - inspects declared type of an entity or the type and value category of an expression  

### `std::iota`  
Defined in header `<numeric>`  
```  
template<class ForwardIterator, class T>  
void iota(ForwardIterator first, ForwardIterator last, T value);  
```  

