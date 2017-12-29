# `struct`s, `union`, `enum`  

## Declaring a variable of a type of `struct` to be used globally from the header 

cf. `./structs/structs.h`  

cf. [4.2 — Global variables and linkage](http://www.learncpp.com/cpp-tutorial/42-global-variables/)  

### Internal and external linkage via the static and extern keywords ###

In addition to scope and duration, variables have a third property: *linkage.*   
A variable’s **linkage** determines whether multiple instances of an identifier refer to the same variable or not.  

A variable with internal linkage is called an **internal variable** (or static variable).  
Variables with internal linkage can be used anywhere within the file they are defined in, but can not be referenced outside the file they exist in.

## `Plain Old Data`  

cf. 8.2.6 Plain Old Data from Ch. 8 Structures, Unions, and Enumerations, pp. 210 of Bjarne Stroustrup, The C++ Programming Language, 4th Ed.  

Sometimes want to treat object as "plain old data" (contiguous sequence of bytes in memory) and not worry about advanced semantic notions such as 
- run-time polymorphism (Secs. 3.2.3, 20.3.2)  
- user-defined copy semantics (Secs. 3.3, 17.5)  
- etc.  
because, so to be able to move objects around efficiently, depending on hardware.  

e.g.  

```  
struct S0 {};	// a POD
struct S1 {int a; }; // a POD
struct S2 {
	int a;
	S2(int aa) : a(aa) {} };	// not a POD (no default constructor)  
struct S3 { 
	int a;
	S3(int aa) : a(aa) { }  
	S3() {} }; // a POD (user-defined default constructor  
struct S4 { int a; 
	S4(int aa) : a(aa) { }  
	S4() = default; }; // a POD
struct S5 { virtual void f(); /* ... */ }; // not a POD (has a virtual function)  	

struct S6 : S1 {}; 		// a POD
struct S7 : S0 { int b; }; // a POD  
struct S8 : S1 { int b; }; // not a POD (data in both S1 and S8) 
struct S9 : S0, S1 {} ; // a POD

```  

To manipulate an object as a POD ("just data"), object must 
* not have complicated layout (e.g. with **vptr** (Sec. 3.2.3, 20.3.2)  
* not have nonstandard (user-defined) copy semantics, and 
* have trivial default constructor  

### POD precise definition 
POD object must be of 
* *standard layout type*  
* *trivially copyable type*  
* type with trivial default constructor  

Def. of *trivial type*, type with  
* trivial default constructor (e.g. use `=default` (Sec. 17.6.1) and 
* trivial copy and move operations  

type has standard layout unless it  
* has non-**static** member or base that's not standard layout  
* has **virtual** function (Sec. 3.2.3, Sec. 20.3.2)  
* has **virtual** base  (Sec. 21.3.5)  
* has member that's reference (Sec. 7.7)  
* has multiple access specifiers for non-static data members (Sec. 20.5), or 
* prevents important layout optimizations  
	* by having non-**static** data members in more than 1 base class or in both derived class and base, or  
	* by having base class of same type as 1st non-**static** data member  

Basically, standard layout type is 1 that has layout with obvious equivalent in C and is in union of what common C++ Application Binary Interfaces (ABIs) can handle.  

Type is trivially copyable unless it has nontrivial copy operation, move operation or destructor (3.2.1.2, 17.6)  
copy, move, or destructor nontrivial if  
* it's user-defined  
* its class has **virtual** function  
* its class has **virtual** base  
* its class has base or member that's not trivial  

Object of built-in type is trivially copyable, and has standard layout;   
also, array of trivially copyable objects is trivially copyable; array of standard layout objects has standard layout.  

**`is_pod`**  is standard-library property predicate (35.4.1) defined in `<type_traits>` allowing us to ask question "Is `T` a POD?"   

	
## Fields  

cf. 8.2.7 Fields from Ch. 8 Structures, Unions, and Enumerations, pp. 212 of Bjarne Stroustrup, The C++ Programming Language, 4th Ed.  

field, also called *bit-field*, are convenient shorthand for using bitwise logical operators (Sec. 11.1.1) to extract information from and insert information into part of a word.   
not possible to take address of a field  

### Fields and `struct`s  

bundle several fields, such tiny variables together, in a `struct`  
