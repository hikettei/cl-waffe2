
#pragma SIMD
#pragma GCC optimize ("O3")
#pragma GCC target ("avx2")

#include <math.h>
#include <stdio.h>
#include <stdint.h>

#if defined(__x86_64)
  #include "simd/x86_64.h"
#elif defined(__aarch64__)
  #include "simd/aarch64.h"
#endif

// TODO:
// Arithmetic Ops +-/*
// Scalar Arithmetic Ops
// Mathematical Ops with SIMD (Powered by SLEEF?)
// Compare Ops
// Hardware specific optimization for this:
// Unfold for CNN, im2col.
// If possible, more sparse matrix support, especially, sparse gemm

// Load
waffe2_svec static inline make_waffe2_svec (float* x, const long incx) {
  waffe2_svec out;
  for (int i=0;i<SIMD_SINGLE_STRIDE;i++) out[i] = x[i*incx];
  return out;
}

waffe2_dvec static inline make_waffe2_dvec (double* x, const long incx) {
  waffe2_dvec out;
  for (int i=0;i<SIMD_DOUBLE_STRIDE;i++) out[i] = x[i*incx];
  return out;
}

void static inline strided_waffe2_store_svec(float* ptr, waffe2_svec x, const long incx) {
  for (int i=0; i<SIMD_SINGLE_STRIDE;i++) (ptr + i*incx)[0] = x[i];
}

void static inline strided_waffe2_store_dvec(double* ptr, waffe2_dvec x, const long incx) {
  for (int i=0; i<SIMD_DOUBLE_STRIDE;i++) (ptr + i*incx)[0] = x[i];
}

// Store
// x <- vec

// Memo: __builtin_assume_aligned

// two_arg_f(n, x*, x_stride, y*, y_stride)
// y <- op(x, y)
// waffe2_sadd
// define_arithmetic_func(sadd, SIMD_SINGLE_FLOAT, __mm256, prefix, float, simd_op_name, +)
#define define_arithmetic_func(define_as, stride, prefix, dtype, simd_op_name, reminder_op_name) \
  void waffe2_##define_as(const long n, dtype* x, const long incx, dtype* y, const long incy) \
  {									\
    dtype *y_end = y + n * incy;					\
    dtype *y_simd_end = y + (n/stride)*stride;				\
    waffe2_##prefix##vec vx, vy;					\
    if (incx == 1 && incy == 1)						\
      {									\
	while (y != y_simd_end)						\
	  {								\
	    vx = waffe2_load_##prefix##vec(x);				\
	    vy = waffe2_load_##prefix##vec(y);				\
	    vy = simd_op_name(vx, vy);					\
	    waffe2_store_##prefix##vec(y, vy);				\
	    x += stride;						\
	    y += stride;						\
	  }								\
      }									\
    else if (incx == 1)							\
      {									\
	while (y != y_simd_end)						\
	  {								\
	    vx = waffe2_load_##prefix##vec(x);  			\
	    vy = make_waffe2_##prefix##vec(y, incy);			\
	    vy = simd_op_name(vx, vy);					\
	    strided_waffe2_store_##prefix##vec(y, vy, incy);	       	\
	    x += stride;						\
	    y += stride;						\
	  }								\
      }									\
    else if (incy == 1)							\
      {									\
	while (y != y_simd_end)						\
	  {								\
	    vy = waffe2_load_##prefix##vec(y);  			\
	    vx = make_waffe2_##prefix##vec(x, incx);			\
	    vy = simd_op_name(vx, vy);					\
	    waffe2_store_##prefix##vec(y, vy);				\
	    x += stride;						\
	    y += stride;						\
	  }								\
      }									\
    else								\
      {									\
	while (y != y_simd_end)						\
	  {								\
	    vx = make_waffe2_##prefix##vec(x, incx);			\
	    vy = make_waffe2_##prefix##vec(y, incy);			\
	    vy = simd_op_name(vx, vy);					\
	    strided_waffe2_store_##prefix##vec(y, vy, incy);	       	\
	    x += stride;						\
	    y += stride;						\
	  }								\
      }									\
    while (y != y_end) {						\
      y[0] = reminder_op_name(x[0], y[0]);				\
      x += incx;							\
      y += incy;							\
    }									\
    };									

float static inline sas_sadd(float x, float y) { return x + y; }
float static inline sas_ssub(float x, float y) { return x - y; }
float static inline sas_smul(float x, float y) { return x * y; }
float static inline sas_sdiv(float x, float y) { return x / y; }

float static inline sas_dadd(double x, double y) { return x + y; }
float static inline sas_dsub(double x, double y) { return x - y; }
float static inline sas_dmul(double x, double y) { return x * y; }
float static inline sas_ddiv(double x, double y) { return x / y; }

//                     DEFINE_AS | STRIDE | PREFIX | DTYPE | SIMD_OP
// waffe2_sadd ...

define_arithmetic_func(sadd, SIMD_SINGLE_STRIDE, s, float, waffe2_simd_sadd, sas_sadd);
define_arithmetic_func(ssub, SIMD_SINGLE_STRIDE, s, float, waffe2_simd_ssub, sas_ssub);
define_arithmetic_func(smul, SIMD_SINGLE_STRIDE, s, float, waffe2_simd_smul, sas_smul);
define_arithmetic_func(sdiv, SIMD_SINGLE_STRIDE, s, float, waffe2_simd_sdiv, sas_sdiv);

define_arithmetic_func(dadd, SIMD_DOUBLE_STRIDE, d, double, waffe2_simd_dadd, sas_dadd);
define_arithmetic_func(dsub, SIMD_DOUBLE_STRIDE, d, double, waffe2_simd_dsub, sas_dsub);
define_arithmetic_func(dmul, SIMD_DOUBLE_STRIDE, d, double, waffe2_simd_dmul, sas_dmul);
define_arithmetic_func(ddiv, SIMD_DOUBLE_STRIDE, d, double, waffe2_simd_ddiv, sas_ddiv);
     
