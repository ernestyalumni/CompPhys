cf. Edward Scheinerman. **C++ for Mathematicians: An Introduction for Students and Professionals.** 1st Edition. *CRC Press*; 1 edition (June 8, 2006). ISBN-13: 978-1584885849

- `./03-gcd/` Greatest Common Denominator  

# Double-inclusion safeguards, preprocessors `#ifndef`, `#define`, `#endif`  

e.g. `./03-gcd/gcd.h`  

Line `#ifndef GCD_H`, is instruction to preprocessor, that stands for "if not defined".  If `GCD_H` not defined, we should do what follows up to matching `#endif`.   

`#define GCD_H`, defines symbol `GCD_H`, although doesn't specify any particular value for symbol (we just want to know whether it's defined).    
`#endif` prevent double inclusion  

e.g. 
```  
#ifndef GCD_H
#define GCD_H  

long gcd(long a, long b);

#endif // END of GCD_H  
```  

# 2 different procedures (functions) may have same name, just have different types of arguments  

e.g. `./03-gcd/gcd-extended.cpp`  


