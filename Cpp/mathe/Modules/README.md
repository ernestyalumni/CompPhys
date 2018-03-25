cf. pp. 448, Ch. 16 **Classes** by Bjarne Stroustrup, **The C++ Programming Language**, *4th Ed.*.  
cf. Edward Scheinerman, C++ for Mathematicians: An Introduction for Students and Professionals. Taylor & Francis Group, 2006. 

cf. pp. 97, Sec. 6.3. Data hiding, Scheinerman (2006).  
The reason for putting data in the private section is to protect those data from tampering. Tampering by whom? You, of course!

cf. pp. 98, Scheinerman (2006)
On classes, and on data hiding,
Here's a mathematical analogy to data hiding. 
In analysis, all we need to know about the real number system is that it's a complete, ordered field. 
2 analysts - Annie and Andy - may have different preferences on how to define the reals. 
Annie prefers Dedekind cuts because these make the proof of the least upper bound property easy. 
Andy, however, prefers equavalence classes of Cauchy sequences, because it's easier to define the field operations. 
In both cases, the analysts' 1st. job is to verify that their system satisfies the complete ordered field axioms.  
From that point on, they can "forget" how they defined real numbers; the rest of their work is identical. 
The "private" data of real numbers is either a Dedekind cut or a Cauchy sequence; we need not worry about which.  
The "public" part of the real numbers are the complete ordered field axioms.  

cf. pp. 100 6.5 Assignment and conversion, Scheinerman (2006)

Constructors serve a 2nd. purpose in C++; they can be used to convert from 1 type to another. 
In the case of `Point` case, we didn't provide a single-argument constructor. However, we can convert a *pair* of `double` values to a `Point` like this:
```
Point P;
....
P = Point(-5.2, 4.2);
```

cf. pp. 106, Scheinerman (2006)

left argument is nameless, when referring to the class object we're working on.  
How can we refer to the left argument (LHS), which is the class object itself, in its entirety?  
Solution is this: Use `*this`.  
The expression `*this` is a reference to the object whose method is being defined.  This enables us to build on the `==` procedure and use it in writing the `!=` procedure.  

`*this` consists of 2 parts: operator `*` and pointer `this`.   
The `this` pointer always points to the location that holds the object whose method is being defined.  
The `*` operator *dereferences* the pointer; this means that `*this` is the object housed at location `this`.  
i.e. `*this` is the object for whome the current method is being applied. 

the object `cout` is of type `ostream` (which stands for *output stream*) and is declared in header `iostream`.  
Therefore `cout << P` contains 2 arguments: the left argument is of type `ostream` and right is of type `Point`.  
Furthermore, the result of this expression is also of type `ostream`.  

```
ostream& operator<<(ostream& os, const Point& P);
```
* return type of this proecdure is `ostream&`.  When we invoke the `<<` operator via the expression `cout << P`, the result of the procedure is `cout`, an object of type `ostream`.  
  - The ampersand indicates this procedure returns a reference to (not a copy of) the result.  
* name of this procedure is `operator<<` 
*  procedure takes 2 orguments; 1st. is left-hand argument to `<<` and 2nd. is taken from right.
  - 1st (left) argument is of type `ostream`. We call this argument `os` for "output stream". The call is by reference, not by value. 
  - Objects of type `ostream` are large and complicated; we don't want to make a copy of them.  
  - Furthermore, operators should be invoked using call by reference. 
  - This argument is not declared `const` because act of printing changes the data held in the `ostream` object. 
  - 2nd. (right) argument is of type `Point`. Again, we call by reference because that's what C++ requires for operators. 
  - Printing a `Point` doesn't affect its data; we certify this by flagging this argument as `const`.  

When we execute statement `cout << P`, in effect we create a call that could be thought of like this: `operator<<(cout, P)`.  

pp. 182, Scheinerman (2006)

* class `Child` declared to be a *public subclass* of class `Parent`  
 - `Parent` class houses 2 `double` real values, `x` and `y`, in `private` 
* declare a private element for `Child`: an integer `k`.  The class `Child` therefore holds 3 data elements: `x`, `y`, and `k`
  - A method in `Child` can access `k` but not `x` or `y`. The latter are private to `Parent` and children have no right to examine their parents' private parts. 
* following code is illegal  
```
Child(double a, double b, int n) {
  x = a; y = b; k = n;
}
```
The problem is that `Child` cannot access `x` or `y`.  
Logically, what we want to do is this: 
1st., we want to invoke the constructor for `Parent` with the arguments `a` and `b`  
2nd., we do additional work special for the `Child` class, namely, assign `n` to `k`.  

Before open brace for the method, we see a colon and call to `Parent(a,b)`. By this syntax, a constructor for a derived class (`Child`) can pass its arguments to the constructor for the base class (`Parent`).  
The general form for constructor of a derived class is 
```
derived_class(argument list) : base_class(argument_sublist)
{
  more stuff to do for the derived class;
}
```  
When derived class's constructor is invoked, it 1st calls its parent's constructor(passing none, some, or all of the arguments up to its parent's constructor).  Once base class's constructor completes its work, the code inside the curly braces executes to do anything extra that's required by the derived class.


