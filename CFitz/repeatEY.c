/* repeatEY.c */
/*
  Fitzpatrick.  Computational Physics.  329.pdf
  2.15 Command Line Parameters pp. 77
  
  EY suffix means, this is my own changes, to explore what char *argv[] means

  "
  The main() function may optionally possess special arguments which allow parameters
  to be passed to this function from the operating system.  There are 2 such arguments,
  conventionally called argc and argv.
  
  argc is an integer which is set to the number of parameters passed to main()
  
  argv is an array of pointers to character strings which contain these parameters

  In order to pass one or more parameters to a C program when it is executed from the operating system, 
  parameters must follow the program name on the command line: e.g.
  
  % program-name parameter_1 parameter_2 parameter_3 .. parameter_n

  Program name will be stored in the first item in argv, followed by each of the parameters. 
  Hence, if program name is followed by n parameters there'll be n + 1 entries in argv,
  ranging from argv[0] to argv[n]
  Furthermore, argc will be automatically set equal to n+1
*/

/*
  Program to read and echo data from command line
*/

#include <stdio.h>

int main(int argc, char *argv[])
{
  int i;
  
  for (i=1; i < argc; i++) printf("%s ", argv[i]);
  printf("\n");

  //  char *argvEY[];  // obtain this error: repeatEY.c:41:9: error: array size missing in ‘argvEY’
  //   char *argvEY[];
  //       ^

  char *argvEY[10];
  //   *argvEY[5] =     { "Cash", "Rules", "everything", "around", "me" }   ; doesn't work
  // error: error: expected expression before ‘{’ token
  //  *argvEY[5] =     { 

  argvEY[0] = "Cash"; argvEY[1] = "Rules"; argvEY[2] = "everything"; argvEY[3] = "around"; argvEY[4] = "me";  

  printf("\n argvEY[1] = %d  &argvEY[1] = %X  argvEY = %X  *argvEY[1] = %d \n", argvEY[1], &argvEY[1], argvEY, *argvEY[1]);

  printf("\n argvEY[1] = %s  &argvEY[1] = %X  argvEY = %X  *argvEY[1] = %X \n", argvEY[1], &argvEY[1], argvEY, *argvEY[1]);

  // The point of this is practicing the concept of an array as a pointer to the first element of a memory block, that deferences to chars.  I found it difficult to understand; I've tried to think of this in terms of category theory to alleviate this (cf. ML.pdf of the MLgrabbag repository, ernestyalumni).  
  
  return 0;
}

