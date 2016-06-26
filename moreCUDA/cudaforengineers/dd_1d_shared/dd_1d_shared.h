/* dd_1d_shared.h
 * 1-dimensional double derivative (dd for '') by finite difference with global memory
 * Ernest Yeung  ernestyalumni@gmail.com
 * 20160625
 */
#ifndef __DD_1D_SHARED_H__
#define __DD_1D_SHARED_H__

void ddParallel(float *out, const float *in, int n, float h);

#endif // __DD_1D_SHARED_H__
