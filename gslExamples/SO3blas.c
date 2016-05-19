/* SO3blas.c */

// SO(3) Lie group implemented with GNU GSL
// cf. wikipedia, "Rotation_group_SO(3)#Infinitesimal_rotations"

// EY note : 20160519 Bizarrely, matrix multiplication isn't implemented
// in a straightforward manner with the gsl_matrix class;
// see blas (which I don't know what it is yet) and dgemm for the libraries

/*
  Compiling advice:
  gcc matrixio.c -lgsl -lgslcblas

 */


#include <stdio.h>
#include <gsl/gsl_cblas.h>
#include <gsl/gsl_blas.h>

// Function prototype
/* int commute(gsl_matrix_view, gsl_matrix_view); */

void commute1(void);


// global variables

double l_x[] = {  0,  0,  0,
		    0,  0, -1,
		    0,  1,  0};

double l_y[] = {  0,  0,  1,
		    0,  0,  0,
		   -1,  0, 0};

double l_z[] = {  0, -1,  0,
		    1,  0,  0,
		    0,  0,  0};
// cannot put gsl_matrix view commands in global; I need to check why:
// error: initializer element is not constant


int main(void)
{
  gsl_matrix_view L_x = gsl_matrix_view_array(l_x,3,3);
  gsl_matrix_view L_y = gsl_matrix_view_array(l_y,3,3);
  gsl_matrix_view L_z = gsl_matrix_view_array(l_z,3,3);

  
  printf("[ %g, %g, %g]\n", l_x[0], l_x[1], l_x[2] );
  printf("[ %g, %g, %g]\n", l_x[3], l_x[4], l_x[5] );
  printf("[ %g, %g, %g]\n\n", l_x[6], l_x[7], l_x[8] );

  printf("[ %g, %g, %g]\n", l_y[0], l_y[1], l_y[2] );
  printf("[ %g, %g, %g]\n", l_y[3], l_y[4], l_y[5] );
  printf("[ %g, %g, %g]\n\n", l_y[6], l_y[7], l_y[8] );

  printf("[ %g, %g, %g]\n", l_z[0], l_z[1], l_z[2] );
  printf("[ %g, %g, %g]\n", l_z[3], l_z[4], l_z[5] );
  printf("[ %g, %g, %g]\n\n", l_z[6], l_z[7], l_z[8] );


  double xy[] = { 0.00, 0.00, 0.00, 
		  0.00, 0.00, 0.00,
		  0.00, 0.00, 0.00 };
  
  gsl_matrix_view XY = gsl_matrix_view_array(xy,3,3);

  gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0,
		 &L_x.matrix,&L_z.matrix,
		 0.0, &XY.matrix);

  printf("[ %g, %g, %g ]\n", xy[0], xy[1], xy[2]);
  printf("[ %g, %g, %g ]\n", xy[3], xy[4], xy[5]);
  printf("[ %g, %g, %g ]\n\n", xy[6], xy[7], xy[8]);

  commute1();
  
  return 0;
}


void commute1(void) {

  gsl_matrix_view L_x = gsl_matrix_view_array(l_x,3,3);
  gsl_matrix_view L_y = gsl_matrix_view_array(l_y,3,3);
  gsl_matrix_view L_z = gsl_matrix_view_array(l_z,3,3);


  double xy[] = { 0.00, 0.00, 0.00, 
		  0.00, 0.00, 0.00,
		  0.00, 0.00, 0.00 };
  
  gsl_matrix_view XY = gsl_matrix_view_array(xy,3,3);

  double yx[] = { 0.00, 0.00, 0.00, 
		  0.00, 0.00, 0.00,
		  0.00, 0.00, 0.00 };
  
  gsl_matrix_view YX = gsl_matrix_view_array(yx,3,3);



  
  gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0,
		 &L_x.matrix,&L_y.matrix,
		 0.0, &XY.matrix);

  gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0,
		 &L_y.matrix,&L_x.matrix,
		 0.0, &YX.matrix);

   printf("[ %g, %g, %g ]\n", yx[0], yx[1], yx[2]);
   printf("[ %g, %g, %g ]\n", yx[3], yx[4], yx[5]);
   printf("[ %g, %g, %g ]\n", yx[6], yx[7], yx[8]);
 
  
}


/*

int commute(gsl_matrix_view X, gsl_matrix_view Y)
{
  double xy[] = { 0.00, 0.00, 0.00, 
		  0.00, 0.00, 0.00,
		  0.00, 0.00, 0.00 };

  double yx[] = { 0.00, 0.00, 0.00, 
		  0.00, 0.00, 0.00,
		  0.00, 0.00, 0.00 };

  double bracket[] = { 0.00, 0.00, 0.00, 
		       0.00, 0.00, 0.00,
		       0.00, 0.00, 0.00 };
  
  gsl_matrix_view XY = gsl_matrix_view_array(xy,3,3);
  gsl_matrix_view YX = gsl_matrix_view_array(yx,3,3);
  gsl_matrix_view Bracket = gsl_matrix_view_array(bracket,3,3); 

  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,1.0,
		 &X.matrix,&Y.matrix,
		 0.0, &XY.matrix);
  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,1.0,
		 &X.matrix,&Y.matrix,
		 0.0, &XY.matrix);
  gls_blas_dgemm(CblasNoTrans, CblasNoTrans,1.0,
		 &XY.matrix, &YX.matrix,
		 0.0, &Bracket.matrix);
  
  return 1;
}

*/
