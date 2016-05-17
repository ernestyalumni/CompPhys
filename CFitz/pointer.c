/* pointer.c */
/*
  Fitzpatrick.  Computational Physics.  329.pdf

  Programming note from Fitzpatrick.  pp.55
  2.10 Pointers
  
  "pass the arguments of a function by reference, rather than by value, 
  using pointers. This allows the two-way communication of information via arguments 
  during function calls." -Fitzpatrick
  
*/

/*
  Simple illustration of the action of pointers
*/

#include <stdio.h>

int main() {
  int u = 5;
  int v;
  int *pu; // Declare pointer to an integer variable
  int *pv; // Declare pointer to an integer variable

  pu = &u; // Assign address of u to pu
  v = *pu; // Assign value of u to v
  pv = &v; // Assign address of v to pv

  printf("\nu = %d  &u = %X  pu = %X  *pu = %d", u, &u, pu, *pu);
  printf("\nv = %d  &v = %X  pv = %X  *pv = %d\n", v, &v, pv, *pv);

  return 0;
}

// conversion character X, appearing in the control strings of the above
// printf() function calls, indicates associated data item should be output as
// hexadecimal number


