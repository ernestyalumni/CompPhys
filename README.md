# CompPhys
CompPhys - a Computational Physics repository

## C Resources

Coursera's Heterogeneous Parallel Programming (taught by Wen-mei W. Hwu) suggested these links to brush up on [C programming](https://class.coursera.org/hetero-004/wiki/Introduction_to_C):
-    http://www.cprogramming.com/tutorial/c-tutorial.html
-    http://www.physics.drexel.edu/courses/Comp_Phys/General/C_basics/
-    http://gd.tuwien.ac.at/languages/c/cref-mleslie/CONTRIB/SAWTELL/intro.html
-    http://en.wikibooks.org/wiki/C_Programming
-    http://www.cprogramming.com/

I also found this pdf from [Fitzpatrick, "Computational Physics"](http://farside.ph.utexas.edu/teaching/329/329.pdf).  It has fully working code.  So I will try to implement some of the code.  

### Pointers (for Computational Physics)

## GNU Scientific Library; GNU GSL 

I am using [GNU Scientific Library](http://www.gnu.org/software/gsl/manual/gsl-ref.pdf) (GNU GSL) because there are some serious concerns with [Numerical Reciples](http://www.lysator.liu.se/c/num-recipes-in-c.html): cf. https://www.reddit.com/r/Physics/comments/s9p16/the_numerical_recipes_license_is_the_riaa_of_the/ 

The rationale for GNU GSL is [clear and fairly straightforward](https://www.gnu.org/software/gsl/design/gsl-design.html).

### Compiling your C code using GSL, and making the executable to run in GSL

The [GNU GSL manual](http://www.gnu.org/software/gsl/manual/gsl-ref.pdf) is relatively straightforward and clear about the hoops you'll have to jump through to compile your C code that uses GSL, and making the executable so you can get a result.  For instance, on a Mac OS X, I typed up the example program on pp. 4, Section 2.1 (it's saved as `gsl_bessel_example.c` in this repository) and I ran these 2 commands:  
```
	gcc -Wall -I/usr/local/include -c gsl_bessel_example.c  
      	gcc -L/usr/local/lib example.o -lgsl -lgslcblas -lm  
```    

The manual explains thoroughly and understandably what the flags in the command mean.  Note that when I removed the `-c` flag in the first command, hoping to not only create the `.o` object file, but to automatically make the executable, I was greeted with an error (!!!).  

```   
      ld: symbol(s) not found for architecture x86_64  
      clang: error: linker command failed with exit code 1 (use -v to see invocation)  
```    



 

