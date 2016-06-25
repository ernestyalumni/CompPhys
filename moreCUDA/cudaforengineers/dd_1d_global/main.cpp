/* main.cpp
 * 1-dimensional double derivative ('') by finite difference with global memory
 * Ernest Yeung  ernestyalumni@gmail.com
 * 20160625
 */
 #include "dd_1d_global.h"
 #include <math.h>
 #include <stdio.h>
 
int main() {
	const float PI = acos(-1.);
	const int N = 150;
	const float h = 2 * PI / N;
	 
	float x[N] = { 0.0 };
	float u[N] = { 0.0 };
	float result_parallel[N] = { 0.0 };
	 
	for (int i = 0; i < N; ++i) {
		x[i] = 2*PI*i / N;
		u[i] = sinf(x[i]);
	}
	
	ddParallel(result_parallel, u, N, h);
	
	FILE *outfile = fopen("results.csv", "w");
	for (int i = 1; i < N-1; ++i) {
		fprintf(outfile, "%f,%f,%f,%f\n", x[i], u[i],
				result_parallel[i], result_parallel[i]+u[i]);
	}
	fclose(outfile);
}
