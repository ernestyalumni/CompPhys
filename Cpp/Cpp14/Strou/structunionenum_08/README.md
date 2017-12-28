# `struct`s, `union`, `enum`  

## Declaring a variable of a type of `struct` to be used globally from the header 

cf. `./structs/structs.h`  

cf. [4.2 — Global variables and linkage](http://www.learncpp.com/cpp-tutorial/42-global-variables/)  

### Internal and external linkage via the static and extern keywords ###

In addition to scope and duration, variables have a third property: *linkage.*   
A variable’s **linkage** determines whether multiple instances of an identifier refer to the same variable or not.  

A variable with internal linkage is called an **internal variable** (or static variable).  
Variables with internal linkage can be used anywhere within the file they are defined in, but can not be referenced outside the file they exist in.
