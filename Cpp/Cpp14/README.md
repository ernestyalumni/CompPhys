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
| `ptrs_unique_shared.cpp` | `.\` | | |

### Dynamic Memory; smart pointers, `shared_ptr`    
cf. Stanley B. Lippman, Josée Lajoie, Barbara E. Moo.  **C++ Primer** (5th Edition) 5th Edition.  *Addison-Wesley Professional*; 5 edition (August 16, 2012).  ISBN-13: 978-0321714114.  

cf. Ch. 12. Dynamic Memory.  

"So far", prior, or prior to C++11/14, objects had well-defined lifetimes.  
- global objects allocated at program startup and destroyed when program ends  
- local, automatic objects are created and destroyed when block they're defined in is entered and exited.  
- local `static` objects allocated before their 1st use and destroyed when program ends.  

Dynamically allocated objects have lifetime that's independent of where they're created; *they exist until they're explicitly freed.*  

Programs "so far" have used only static or stack memory.  
- static memory used for local `static` objects (cf. Sec. 6.1.1, p. 205 of Lippman, Lajoie, Moo (2012)), for class `static` data members (cf. Sec. 7.6, p. 300 of Lippman, Lajoie, Moo (2012)), and for variables defined outside any function  
    * `static` objects allocated before they're used; destroyed when program ends  
- stack memory used for nonstatic objects defined inside functions  
    * stack objects exist only while block in which they're defined is executing  
- Objects allocated in static or stack memory are automatically created and destroyed by compiler  

#### **heap** or **free store**  
**heap** or **free store** - program has a pool of memory it can use and programs use the heap for objects they **dynamically allocate**, i.e. objects that program allocates at run time.  
    - program controls lifetime of dynamic objects; code must explicitly destroy such objects when they're no longer needed.  

cf. 12.1. Dynamic Memory and Smart Pointers of Lippman, Lajoie, Moo (2012)  

#### `new` and `delete`; Problems with them  

In C++, dynamic memory is managed by operators `new` and `delete`  
- `new` allocates, and optionally initializes, object in dynamic memory, and returns pointer to that object  
- `delete` takes a pointer to a dynamic object, destroy object, and free associated memory

Problems: 
- forget to free memory -> memory leak  
- free memory when there are still pointers referring to that memory, in which case we have a pointer that refers to memory that's no longer valid.  

To make dynamic memory easier (and safer),  
in `memory` header,  
- **`shared_ptr`** - allows multiple pointers to refer to the same object  
- **`unique_ptr`** - which "owns" object to which it points 
- **`weak_ptr`** - weak reference to object managed by `shared_ptr`
 
#### `make_shared`  

Safest way to allocate and use dynamic memory is to call library function named **`make_shared`** - `make_shared` allocates and initializes an object in dynamic memory and returns `shared_ptr` that points to that object  

### How do I make a smart pointer, `shared_ptr` or `unique_ptr` to an array?  I want this array to be dynamically allocated.   

cf. [`shared_ptr` to an array : should it be used?](https://stackoverflow.com/questions/13061979/shared-ptr-to-an-array-should-it-be-used) 

See `sh_ptr.cpp`  
