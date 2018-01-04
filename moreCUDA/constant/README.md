# on constant memory, `__constant__` in CUDA  

cf. [F.3.3.1. Device Memory Space Specifiers](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-specifiers)  

`__device__`, `__shared__`, `__constant__` memory space specified are *not allowed* on:  
* `class`, `struct`, and `union` data members,  
* formal parameters (???)  
* local variables within a function that executes on the host  

`__shared__`, `__constant__` variables have implied static storage e.g. [What does "static" mean in C](https://stackoverflow.com/questions/572547/what-does-static-mean-in-c), so it's static variable that keeps its value between invocations, and as static global variable, is "seen" only in file it's declared, and only once  

`__device__`, `__constant__`, `__shared__` variables defined in namespace scope, that are of class type, *can't have non-empty constructor or non-empty destructor* (so it must have empty constructor and empty destructor?)  
`__device__`, `__constant__`, `__shared__` *must have empty constructor and empty destructor*.  
Constructor for class type considered empty at point in translation unit, if it's either a trivial constructor or it satisfies all following conditions:  
* constructor function has been defined  
* constructor function has no parameters, initializer list empty, and function body is empty compound statement  
* class has no virtual functions and no virtual base classes  
* default constructors of all base classes of its class can be considered empty.  
* for all nonstatic data members of its class that are of class type (or array thereof), default constructors can be considered empty  

Destructor for class considered empty at point in translation unit, if it's either trivial destructor or it satisfies all of the following conditions  
* destructor function has been defined
* destructor function body is empty compound statement.  
* class has no virtual functions and no virtual base classes
* destructors of all base classes of its class can be considered empty
* for all non static data members of its class that are of class type (or array thereof), destructor can be considered empty  


 

