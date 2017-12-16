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

Please see [`CompPhys/Cpp/README.md`](https://github.com/ernestyalumni/CompPhys/blob/master/Cpp/README.md) for my original implementations of File I/O, especially the pure C++ implementations of File I/O for writing `.csv` files.  I will try to summarize and make concise those exact same contents (duplicate) here as well, if necessary.  

### File I/O with binary format; and empirically, C++11/C++14 is the way to go  

[PanicSheep](https://stackoverflow.com/users/1112378/panicsheep) did great work in empirically comparing 3 cases of writing to binary.  The gist of the code is the following:  

cf. [Writing a binary file in C++ very fast, stackoverflow](https://stackoverflow.com/questions/11563963/writing-a-binary-file-in-c-very-fast)

```  
long long option_1(std::size_t bytes) 
{
	std::vector<uint64_t> data = GenerateData(bytes);
	
	auto myfile = std::fstream("file.binary", std::ios::out | std::ios::binary);
	myfile.write((char*)&data[0],bytes);
	myfile.close();
}

long long option_2(std::size_t bytes)
{
	std::vector<uint64_t> data = GenerateData(bytes);
	
	FILE* file = fopen("file.binary", "wb");
	fwrite(&data[0], 1, bytes, file);
	fclose(file);
}
	
long long option_3(std::size_t bytes)
{
	std::vector<uint64_t> data = GenerateData(bytes);
	
	std::ios_base::sync_with_stdio(false);
	auto myfile = std::fstream("file.binary", std::ios::out | std::ios::binary);
	myfile.write((char*)&data[0], bytes);
	myfile.close();
}
```  

`std::ios_base::sync_with_stdio` sets whether standard C++ streams are synchronized to the standard C streams after each input/output operation.  

The standard C streams are `stdin, stdout` and `stderr`.  

In practice, synchronized C++ streams are unbuffered, and each I/O operation on a C++ stream is immediately applied to the corresponding C stream's buffer.  This makes it possible to freely mix C++ and C I/O.  

Also, synchronized C++ streams are guaranteed to be thread-safe (individual characters output from multiple threads may interleave, but no data races occur).  

If synchronization is turned off, the C++ standard streams are allowed to buffer their I/O independently, which may be considerably faster in some cases.  

cf. [`std::ios_base::sync_with_stdio` (`cppreference.com`)](http://en.cppreference.com/w/cpp/io/ios_base/sync_with_stdio)  

In 1 run, I obtained the following, on a Desktop with Intel Xeon CPU E5-1650 v3 @ 3.50 GHz * 12, Fedora Workstation 25, g++ v.6.4.1 (CUDA 9 doesn't play well with gcc 7, yet; (20171014)), with `-std=c++14`.  

```  
option1, 1MB: 0ms
option1, 2MB: 1ms
option1, 4MB: 2ms
option1, 8MB: 4ms
option1, 16MB: 8ms
option1, 32MB: 17ms
option1, 64MB: 33ms
option1, 128MB: 67ms
option1, 256MB: 579ms
option1, 512MB: 2180ms
option1, 1024MB: 5160ms
option1, 2048MB: 11247ms
option1, 4096MB: 24994ms
option2, 1MB: 752ms
option2, 2MB: 39ms
option2, 4MB: 2ms
option2, 8MB: 4ms
option2, 16MB: 8ms
option2, 32MB: 17ms
option2, 64MB: 34ms
option2, 128MB: 67ms
option2, 256MB: 697ms
option2, 512MB: 2128ms
option2, 1024MB: 5867ms
option2, 2048MB: 11980ms
option2, 4096MB: 24799ms
option3, 1MB: 539ms
option3, 2MB: 82ms
option3, 4MB: 2ms
option3, 8MB: 5ms
option3, 16MB: 8ms
option3, 32MB: 17ms
option3, 64MB: 34ms
option3, 128MB: 70ms
option3, 256MB: 946ms
option3, 512MB: 2382ms
option3, 1024MB: 5343ms
option3, 2048MB: 12616ms
option3, 4096MB: 25169ms
```  


cf. [Meaning of "const" last in a C++ method declaration? ](https://stackoverflow.com/questions/751681/meaning-of-const-last-in-a-c-method-declaration) 

When you add const keyword to a method, the this pointer will essentially become const, and therefore you can't change any member data (unless use mutable).  

e.g.  
```  
class MyClass1
{
	private:
		mutable int counter;
	public:
		MyClass1() : counter(0) {}
		
		void Foo() {
			counter++;
			std::cout << "Foo" << std::endl;
		}
		
		void Foo() const {
			counter++;
			std::cout << "Foo const" << std::endl;
		}
		
		int GetInvocations() const
		{
			return counter;
		}
};  
```  
 
cf. ["&" meaning after variable type, means you're passing the variable by reference](https://stackoverflow.com/questions/11604190/meaning-after-variable-type)  
  
It means you're passing the variable by reference, i.e. The & means function accepts *address* (or reference) to a variable, instead of *value* of the variable.  

e.g. 
```  
int x = 42;
int& y = x;

MyClass cc;
MyClass & ccc=cc;
const MyClass& c=cc;
```  

## C++11 vs. C++14  

[The C++14 Standard: What You Need to Know, Dr. Dobb's](http://www.drdobbs.com/cpp/the-c14-standard-what-you-need-to-know/240169034)

compiler deduces what type.   

Reasons for *return type deduction*  
1. Use `auto` function return type to return complex type, such as iterator, 
2. refactor code   


## Initializer list 

From Ch. 3, pp. 49, Item 7: Distinguish between () and {} when creating objects, in Meyers  

It's important to distinguish initialization from assignment, with user-defined types:  
```  
Widget w1; // call default constructor
Widget w2 = w1; 	// not an assignment; calls copy ctor ctor=constructor
w1 = w2; 			// an assignment; calls copy operator= 
```  

C++11 introduces *uniform initialization*, single initialization syntax that can, at least in concept, be used anywhere and express everything, based on braces; Meyers calls it "*braced initialization*".    

Braced initialization prohibits implicit *narrowing conversions*:
``` 
double x,y,z;

int sum1{x+y+z}; // error! sum of doubles may not be expressible as int 
```  

### C++'s *most vexing parse* - anything can be parsed as a declaration, must be interpreted as 1  

```  
Widget w1(10); // call Widget ctor=constructor with argument 10, get an object

Widget w2(); 	// most vexing parse!  declares a function named w2 that returns a Widget!  

Widget w3{};	// calls Widget ctor=constructor with no arguments, get an object
```

Another example, 
```  
class Widget {
	public:
		Widget();		// default ctor=constructor
		
		Widget(std::initializer_list<int> il);	// std::initializer
												// _list ctor=constructor
												// no implicit conversion functions
};

Widget w1;	// calls default ctor=constructor
Widget w2{};	// also calls default ctor=constructor 
Widget w3();	// most vexing parse! declares a function!  
}  
```  



### Useful links for initializer list 

cf. [`cplusplus.com` Initializer list](http://www.cplusplus.com/reference/initializer_list/initializer_list/)

Scott Meyers.  Effective Modern C++ 42 Specific Ways  


## [The rule of 3/5/0](http://en.cppreference.com/w/cpp/language/rule_of_three); user-defined destructor, copy constructor, copy assignment/ move constructor, move assignment  

### why Rule of 5  

Because of presence of user-defined destructor, copy-constructor, or copy-assignment operator prevents implicit definition of the move constructor, and move assignment operator  


## copy constructor  

cf. [Copy constructors, cppreference.com](http://en.cppreference.com/w/cpp/language/copy_constructor)  

copy constructor of class T is non-template constructor whose 1st parameter is `T&`, `const T&`, `volatile T&`, or `const volatile T&`.  

**Syntax**  

```  
class_name ( const class_name & )  
class_name ( const class_name & ) = default;  
class_name ( const class_name & ) = delete;
```  

**Explanation**  
1. Typical declaration of a copy constructor.  
2. Forcing copy constructor to be generated by the compiler.  
3. Avoiding implicit generation of copy constructor.  

Copy constructor called whenever an object is **initialized** (by **direct-initialization** or **copy-initialization**) from another object of same type (unless **overload resolution** selects better match or call is **elided** (???)), which includes  
* initialization `T a = b;` or `T a(b);`, where b is of type T;  
* function argument passing: `f(a);`, where a is of type T and f is `void f(T t)`;  
* function return: `return a;` inside function such as `T f()`, where a is of type T, which has no **move constructor**.  

**Example**  

```  
struct A
{
    int n;
    A(int n = 1) : n(n) { }
    A(const A& a) : n(a.n) { } // user-defined copy ctor
};
 
struct B : A
{
    // implicit default ctor B::B()
    // implicit copy ctor B::B(const B&)
};
  
int main()
{
    A a1(7);
    A a2(a1); // calls the copy ctor
    B b;
    B b2 = b;
    A a3 = b; // conversion to A& and copy ctor  
}
```     

i.e. cf. [Copy Constructor in C++](http://www.geeksforgeeks.org/copy-constructor-in-cpp/)

Copy constructor is a member function which initializes an object using another object of the same class. 

### When is copy constructor called?  

1. When object of class returned by value 
2. When object of class is passed (to a function) by value as an *argument*.  
3. When object is constructed based on another object of same class  (or overloaded)  
4. When compiler generates temporary object  

However, it's not guaranteed copy constructor will be called in all cases, because C++ standard allows compiler to optimize the copy away in certain cases.  

### When is used defined copy constructor needed?  shallow copy, deep copy  

If we don't define our own copy constructor, C++ compiler creates default copy constructor which does member-wise copy between objects.  

We need to define our own copy constructor only if an object has pointers or any run-time allocation of resource like file handle, network connection, etc.  

#### Default constructor does only shallow copy.  

#### Deep copy is possible only with user-defined copy constructor.  

We thus make sure pointers (or references) of copied object point to new memory locations.  

### Copy constructor vs. Assignment Operator  

```  
MyClass t1, t2; 
MyClass t3 = t1; 	// ----> (1)
t2 = t1; 			// -----> (2)
```  

Copy constructor called when new object created from an existing object, as copy of existing object, in (1).  
Assignment operator called when already initialized object is assigned a new value from another existing object, as assignment operator is called in (2).  

### Why argument to a copy constructor should be const?   

cf. [Why copy constructor argument should be const in C++?, `geeksforgeeks.org`](http://www.geeksforgeeks.org/copy-constructor-argument-const/)

1. Use `const` in C++ whenever possible so objects aren't accidentally modified.  
2. e.g.  
```  
#include <iostream>  

class Test
{
	/* Class data members */
	public:
		Test(Test &t) 	{ /* Copy data members from t */ } 
		Test()			{ /* Initialize data members */ }
};

Test fun() 
{
	Test t;
	return t;
};

int main()
{
	Test t1;
	Test t2 = fun(); // error: invalid initialization of non-const reference of type ‘Test&’ from an rvalue of type ‘Test’
}

``` 

`fun()` returns by value, so compiler creates temporary object which is copied to t2 using copy constructor (because this temporary object is passed as argument to copy constructor since compiler generates temp. object).  
Compiler error is because *compiler-created temporary objects cannot be bound to non-const references*. 



### [move constructors](http://en.cppreference.com/w/cpp/language/move_constructor) 

cf. [`./movconstruct.cpp`](https://github.com/ernestyalumni/CompPhys/blob/master/Cpp/Cpp14/moveconstruct.cpp)

move constructor of class `T` (or `cls` is my notation), is non-template constructor whose 1st parameter is `T&&`, `const T&&`, `volatile T&&`, or `const volatile T&&`.  and either there are no parameters, or rest of parameters all have default values.  (i.e. `cls&&`, `const cls&&`, `volatile cls&&`, `const volatile cls&&`), i.e. 

For a class, to control what happens when we move, or move and assign object of this class type, use special member function *move constructor*, *move-assignment operator*, and define these operations.  Move constructor and move-assignment operator take a (usually nonconst) rvalue reference, to its type.  Typically, move constructor moves data from its parameter into the newly created object.  After move, it must be safe to run the destructor on the given argument.  cf. Ch. 13 of Lippman, Lajole, and Moo (2012) 


**Syntax**  
```  
class_name( class_name && )  
class_name( class_name &&) = default; 
class_name( class_name &&) = delete; 
```  

Move constructor called whenever by overload, which typically occurs when object is initialized from `rvalue` of same type, including  
* initialization: e.g. `T a = std::move(b);` or `T a(std::move(b));` 
* function argument passing: `f(std::move(a));`  
* function return: `return a;`  

e.g. [`movconstruct.cpp`](https://github.com/ernestyalumni/CompPhys/blob/master/Cpp/Cpp14/moveconstruct.cpp), notes: 

```  
struct A 
{
	std::string s;
	// struct constructor 
	A() : s("test") { }
	// copy constructor
	A(const A& o) : s(o.s) { std::cout << "move failed!\n"; } 
	// move constructor
	A(A&& o) noexcept : s(std::move(o.s)) { }
}; 


struct B : A
{
	std::string s2;
	int n;
	// implicit move constructor B::(B&&) 
	// calls A's move constructor 
	// calls s2's move constructor
	// and makes a bitwise copy of n 
}; 

struct C : B 
{
	~C() { } 	// destructor prevents implicit move constructor C::(C&&) 
}; 

C c1;
C c2 = std::move(c1); // calls copy constructor 
// fails to move for c1 -> c2!
```   

`C`, when moving it, fails, because user-defined destructor (almost always case with classes) prevents implicit move constructor. Solution is to have a user-defined move constructor.    


```  
struct D : B 
{
	D() {  }
	~D() {  }			// destructor would prevent implicit move constructor D::(D&&) 
	D(D&&) = default; 	// forces a move constructor anyways 
}; 
```  
cf. Scott Meyers. **Effective Modern C++**.  pp. 115, Item 17  

**Move constructor** and **move assignment operator**: each performs memberwise moving of non-static data members.  Generated (automatically) only if class contains no user-declared copy operations, move operations, or destructor.  

EY : 20171201 - compiler knows to select the overload with move constructor or generate (automatically) a move constructor?  

See also 

[Move semantics and rvalue references in C++11](https://www.cprogramming.com/c++11/rvalue-references-and-move-semantics-in-c++11.html)

for a very clear exposition about l-values vs. r-values, the contrast (clearly laid out) between C++03 and C++11 in regards to this, and thus the motivation for move constructor.  


## pImpl - pointer to Implementation; shallow copy, deep copy

cf. Item 22: "When using the Pimpl Idiom, define special member functions in the implementation file," pp. 147 of Meyers (2014)   

``` 
class Widget { 			// still in header "widget.h" 
	public:
		Widget();
		~Widget();		// dtor is needed-see below 
		... 
	
	private:
		struct Impl;	// declare implementation struct 
		Impl *pImpl;	// and pointer to it
};
```  

Because `Widget` no longer mentions types \verb|std::string, std::vector|, and `Gadget, Widget` clients no longer need to `#include` headers for these types.  That speeds compilation.  

*incomplete type* is a type that has been declared, but not defined, e.g. `Widget::Impl`.  There are very few things you can do with an incomplete type, but declaring a pointer to it is 1 of them.  

`std::unique_ptr`s is advertised as supporting incomplete types.  But, when `Widget w;`, `w`, is destroyed (e.g. goes out of scope), destructor is called and if in class definition using `std::unique_ptr`, we didn't declare destructor, compiler generates destructor, and so compiler inserts code to call destructor for `Widget`'s data member `m_Impl` (or `pImpl`).  

`m_Impl` (or `pImpl`) is a `std::unique_ptr<Widget::Impl>`, i.e., a `std::unique_ptr` using default deleter. The default deleter is a function that uses `delete` on raw pointer inside the `std::unique_ptr`.  Prior to using `delete`, however, implementations typically have default deleter employ C++11's `static_assert` to ensure that raw pointer doesn't point to an incomplete type.  When compiler generates code for the destruction of the `Widget w`, then, it generally encounters a `static_assert` that fails, and that's usually what leads to the error message.  

To fix the problem, you need to make sure that at point where code to destroy `std::unique_ptr<Widget::Impl>` is generated, `Widget::Impl` is a complete type.  The type becomes complete when its definition has been seen, and `Widget::Impl` is defined inside `widget.cpp`.  For successful compilation, have compiler see body of `Widget`'s destructor (i.e. place where compiler will generate code to destroy the `std::unique_ptr` data member) only inside `widget.cpp` after `Widget::Impl` has been defined.  

For compiler-generated move assignment operator, move assignment operator needs to destroy object pointed to by `m_Impl` (or `pImpl`) before reassigning it, but in the `Widget` header file, `m_Impl` (or `pImpl`) points to an incomplete type.  Situation is different for move constructor.  Problem there is that compilers typically generate code to destroy `pImpl` in the event that an exception arises inside the move constructor, and destroying `pImpl` requires `Impl` be complete.  

Because problem is same as before, so is the fix - *move definition of move operations into the implementation file*.  

For copying data members, support copy operations by writing these functions ourselves, because (1) compilers won't generate copy operations for classes with move-only types like `std::unique_ptr` and (2) even if they did, generated functions would copy only the `std::unique_ptr` (i.e. perform a *shallow copy*), and we want to copy what the pointer points to (i.e., perform a *deep copy*).  

If we use `std::shared_ptr`, there'd be no need to declare destructor in `Widget`.  

Difference stems from differing ways smart pointers support custom deleters.  For `std::unique_ptr`, type of deleter is part of type of smart pointer, and this makes it possible for compilers to generate smaller runtime data structures and faster runtime code.  A consequence of this greater efficiency is that pointed-to types must be complete when compiler-generated special functions (e.g. destructors or move operations) are used.  For `std::shared_ptr`, type of deleter is not part of the type of smart pointer.  This necessitates larger runtime data structures and somewhat slower code, but pointed-to types need not be complete when compiler-generated special functions are employed.  
   

## Glossary/Dictionary/Quick Look Up  

### `std::future`  

**`future`** - Object that can retrieve value from some provider object or function, properly synchronizing this access if in different threads, i.e.  
class template `std::future` provides mechanism to access result of asynchronous operations, e.g. created via `std::async`, `std::packaged_task`, `std::promise`)      

Defined in header `<future>`

```  		
template< class T > class future;			(1) 	(since C++11)
template< class T > class future<T&>;		(2) 	(since C++11)
template<>          class future<void>;		(3) 	(since C++11)
```  


`noexcept` specifier - specifies whether a function will throw exceptions or not   

**Syntax**  
  
**`noexcept`**  				 (1)
**`noexcept`**( *expression* )	 (2)  
  
1. Same as `noexcept ( true ) `  
2. If *expression* evaluates to `true`, function is declared to not throw any exceptions.  



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

### `decltype` - extract type from variable so `decltype` is sort of an operator that evaluates type of passed expression  

cf. [Type Inference in C++ (auto and decltype), geeksforgeeks](http://www.geeksforgeeks.org/type-inference-in-c-auto-and-decltype/)

`decltype` - `decltype` specifier (since C++11) - inspects declared type of an entity or the type and value category of an expression  



### `std::iota`  
Defined in header `<numeric>`  
```  
template<class ForwardIterator, class T>  
void iota(ForwardIterator first, ForwardIterator last, T value);  
```  

### lambda functions  

cf. [Lambda expression in C++, geeksforgeeks.com](http://www.geeksforgeeks.org/lambda-expression-in-c/)

```  
[ capture clause ] (parameters) -> return-type 
{
	definition of method
}
```  
Generally, return-type in lambda expression evaluated by compiler itself, and we don't need to specify that explicitly and -> return-type part can be ignored,  
but in some complex case as in conditional statement, compiler can't make out return type and we need to specify that.  

A lambda expression can have more power than an ordinary function by having access to variables from the **enclosing scope**.   
We can "capture" external variables from enclosing scope by 3 ways:   
* Capture by reference  
* Capture by value  
* Capture by both (mixed capture)  

*Syntax* used for capturing variables:  
* `[&]` : capture all external variable by reference  
* `[=]` : capture all external variable by value  
* `[a, &b]` : capture `a` by value and `b` by reference  

A lambda with empty capture clause `[]` can access only those variable which are local to it.  

e.g. `lambdaexp.cpp`  

### `std::optional`  

cf. [`std::optional` in `cppreference.com`](http://en.cppreference.com/w/cpp/utility/optional)

**class template std::optional** manages an optional contained value, i.e. value that may or may not be present

Common use case for `optional` is return value of function that may fail.  `optional` handles expensive-to-construct objects well and is more readable, as intent expressed explicitly. 

### smart pointers  

#### [`std::unique_ptr`](http://en.cppreference.com/w/cpp/memory/unique_ptr)

Defined in header `<memory>`.  

```  
template<  
    class T,  
    class Deleter = std::default_delete<T>  
> class unique_ptr;  
  

template <  
    class T,  
    class Deleter  
> class unique_ptr<T[], Deleter>;  
```  		

`std::unique_ptr` is a smart pointer that owns and manages another object through a pointer and disposes of that object when the `unique_ptr` goes out of scope. 

cf. Scott Meyers. **Effective Modern C++**.  pp. 118, Item 18 Use `std::unique_ptr` for exclusive-ownership resource management.   

`std::unique_ptr` same size as raw pointers, and for most operations (including dereferencing), they execute exactly the same instructions.  This means you can use them even in situations where memory and cycles are tight.  If a raw pointer is small enough and fast enough for you, a `std::unique_ptr` almost certainly is, too.  

`std::unique_ptr` embodies *exclusive ownership* semantics.  A non-null `std::unique_ptr` always owns what it points to  
* Moving `std::unique_ptr` transfers ownership from source pointer to the destination pointer. (Source pointer is set to null).  
* Copying `std::unique_ptr` isn't allowed  
* Upon destruction, a non-null `std::unique_ptr` destroys its resource, `delete` to the raw pointer by default  

Use `decltype` specifier (prefix) when making your own destructor.  By definition, [`decltype`](http://en.cppreference.com/w/cpp/language/decltype) "inspects the declared type of an entity or type and value category of an expression."  But it's "useful when declaring types that are difficult or impossible to declare using standard notation, like lambda-related types or types that depend on template parameters."    

"Converting a `std::unique_ptr` to a `std::shared_ptr` is easy.  (Meyers, pp. 124, Item 18)

I'm thinking you can use `.release` - returns a pointer to the managed object and releases the ownership.   

##### [overhead for `std::unique_ptr`](https://stackoverflow.com/questions/22295665/how-much-is-the-overhead-of-smart-pointers-compared-to-normal-pointers-in-c)   

`std::unique_ptr` has memory overhead if you provide it with some non-trivial deleter.  

`std::unique_ptr` has time overhead only during constructor (if it has to copy provided deleter and/or null-initialize the pointer), and during destructor (to destroy the owned object).  


##### [overhead for `std::shared_ptr`](https://stackoverflow.com/questions/22295665/how-much-is-the-overhead-of-smart-pointers-compared-to-normal-pointers-in-c)   

`std::shared_ptr` always has memory overhead for reference counter, but it's very small.  

`std::shared_ptr` has time overhead in constructor (to create reference counter), in destructor (to decrement reference counter and possibly destroy the object), and assignment operator (to increment reference counter).  
Due to thread-safety guarantees of `std::shared_ptr`, these increments/decrements are atomic, thus adding some more overhead.    

cf. Meyers, pp. 125, Item 19: Use `std::shared_ptr` for shared-ownership resource management.  

An object accessed via `std::shared_ptrs` has its lifetime managed by those pointers through *shared ownership*.  
No specific `std::shared_ptr` owns the object.  Instead, all `std::shared_ptrs` pointing to it collaborate to ensure its destruction at the point where it's no longer needed.  When last `std::shared_ptr` pointing to an object stops pointing there (e.g. because `std::shared_ptr` is destroyed or made to point to a different object), `std::shared_ptr` destroys object it points to.  

A `std::shared_ptr` can tell whether it’s the last one pointing to a resource by consulting the resource’s *reference count*, a value associated with the resource that keeps track of how many `std::shared_ptrs` point to it.  
`std::shared_ptr` constructors increment this count (usually),  
`std::shared_ptr` destructors decrement it, and   
copy assignment operators do both.  
(If `sp1` and `sp2` are `std::shared_ptrs` to different objects, the assignment “`sp1 = sp2;`” modifies `sp1` such that it points to the object pointed to by `sp2`.  
The net effect of the assignment is that the reference count for the object originally pointed to by `sp1` is decremented, while that for the object pointed to by `sp2` is incremented.)  
If a `std::shared_ptr` sees a reference count of zero after performing a decrement, no more `std::shared_ptrs` point to the resource, so the `std::shared_ptr` destroys it. 

* **`std::shared_ptrs` are twice the size of a raw pointer**, because they internally contain a raw pointer to the resource as well as a raw pointer to the resource’s reference count. 
* **Memory for the reference count must be dynamically allocated.** Conceptually, the reference count is associated with the object being pointed to,  
but pointed-to objects know nothing about this.  
They thus have no place to store a reference count. (A pleasant implication is that any object-even those of built-in types-may be managed by `std::shared_ptrs.`)  
Item 21 explains that the cost of the dynamic allocation is avoided when the `std::shared_ptr` is created by `std::make_shared`, but there are situations where `std::make_shared` can’t be used. Either way, the reference count is stored as dynamically allocated data.
* **Increments and decrements of the reference count must be atomic,**  

Move-constructing a `std::shared_ptr` from another `std::shared_ptr` sets the source `std::shared_ptr` to null, and that means that the old `std::shared_ptr` stops pointing to the resource at the moment the new `std::shared_ptr starts`. As a result, no reference count manipulation is required.  Moving `std::shared_ptrs` is therefore faster than copying them: copying requires incrementing the reference count, but moving doesn’t.  


### `std::vector`

#### `std::vector<T>::iterator`  

cf. [`std::vector`, `cppreference.com`](http://en.cppreference.com/w/cpp/container/vector) 

**Member types**

| Member type | Definition |
| =========== | ========== |
| `std::vector<T>::iterator` | `RandomAccessIterator` | 
| `std::vector<T>::const_iterator` | Constant `RandomAccessIterator` | 
| `std::vector<T>::reverse_iterator` | `std::reverse_iterator<iterator>` |
| `std::vector<T>::const_reverse_iterator` | `std::reverse_iterator<const_iterator>` |  
 






