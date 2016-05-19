/* SO3.c */

// SO(3) Lie group implemented with GNU GSL
// cf. wikipedia, "Rotation_group_SO(3)#Infinitesimal_rotations"

// EY note : 20160519 Bizarrely, matrix multiplication isn't implemented in a straightforward manner with the gsl_matrix class; see blas (which I don't know what it is yet) and dgemm for the libraries, and SO3blas.c in this repo for that implementation

#include <stdio.h>
#include <gsl/gsl_matrix.h>

// Function prototype
// gsl_matrix commute(gsl_matrix, gsl_matrix);

int main(void)
{
  gsl_matrix *L_x = gsl_matrix_calloc(3,3);
  gsl_matrix *L_y = gsl_matrix_calloc(3,3);
  gsl_matrix *L_z = gsl_matrix_calloc(3,3);

  gsl_matrix_set(L_x,1,2,-1);
  gsl_matrix_set(L_x,2,1,1);

  gsl_matrix_set(L_y,0,2,1);
  gsl_matrix_set(L_y,2,0,-1);

  gsl_matrix_set(L_z,0,1,-1);
  gsl_matrix_set(L_z,1,0,1);


  for (int i=0;i<3;i++)
    for (int j=0;j<3;j++)
      printf("m(%d,%d) = %g\n", i, j,
	     gsl_matrix_get(L_x,i,j));

  for (int i=0;i<3;i++)
    for (int j=0;j<3;j++)
      printf("m(%d,%d) = %g\n", i, j,
	     gsl_matrix_get(L_y,i,j));

 for (int i=0;i<3;i++)
   for (int j=0;j<3;j++)
      printf("m(%d,%d) = %g\n", i, j,
	     gsl_matrix_get(L_z,i,j));

 
  gsl_matrix_free(L_x);
  gsl_matrix_free(L_y);
  gsl_matrix_free(L_z);

  return 0;
}

//gsl_matrix commute(gsl_matrix x, gsl_matrix y)
//{

//}
