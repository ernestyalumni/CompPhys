/* repeat.c */
/*
  Fitzpatrick.  Computational Physics.  329.pdf
  Command Line

*/

/*
  Program to read and echo data from command line
*/

int main(int argc, char *argv[])
{
  int i;
  
  for (i=1; i < argc; i++) printf("%s ", argv[i]);
  printf("\n");

  return 0;
}

