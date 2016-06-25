/* dd_1d_global.h
 * 1-dimensional double derivative (dd for '') by finite difference with global memory
 * Ernest Yeung  ernestyalumni@gmail.com
 * 20160625
 */
#ifndef __DD_1D_GLOBAL_H__
#define __DD_1D_GLOBAL_H__

void ddParallel(float *out, const float *in, int n, float h);

#endif
