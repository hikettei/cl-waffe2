
#pragma SIMD
#pragma GCC optimize ("O3")
#pragma GCC target ("avx2")

#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <float.h>

#if defined(__x86_64)
  #include "simd/x86_64.h"
#elif defined(__aarch64__)
  #include "simd/aarch64.h"
#endif

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

// TODO:
// Arithmetic Ops +-/*
// Mathematical Ops with SIMD (Powered by SLEEF?)
// Compare Ops
// Hardware specific optimization for this:
// Unfold for CNN, im2col.
// If possible, more sparse matrix support, especially, sparse gemm
//

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

waffe2_ivec static inline make_waffe2_i32vec (int32_t* x, const long incx) {
  waffe2_ivec out;
  for (int i=0;i<SIMD_SINGLE_STRIDE;i++) out[i] = x[i*incx];
  return out;
}

waffe2_ivec static inline make_waffe2_u32vec (uint32_t* x, const long incx) {
  waffe2_ivec out;
  for (int i=0;i<SIMD_SINGLE_STRIDE;i++) out[i] = x[i*incx];
  return out;
}

waffe2_ivec static inline make_waffe2_i16vec (int16_t* x, const long incx) {
  waffe2_ivec out;
  for (int i=0;i<SIMD_SINGLE_STRIDE*2;i++) out[i] = x[i*incx];
  return out;
}

waffe2_ivec static inline make_waffe2_u16vec (uint16_t* x, const long incx) {
  waffe2_ivec out;
  for (int i=0;i<SIMD_SINGLE_STRIDE*2;i++) out[i] = x[i*incx];
  return out;
}

waffe2_ivec static inline make_waffe2_i8vec (int8_t* x, const long incx) {
  waffe2_ivec out;
  for (int i=0;i<SIMD_SINGLE_STRIDE*4;i++) out[i] = x[i*incx];
  return out;
}

waffe2_ivec static inline make_waffe2_u8vec (uint8_t* x, const long incx) {
  waffe2_ivec out;
  for (int i=0;i<SIMD_SINGLE_STRIDE*4;i++) out[i] = x[i*incx];
  return out;
}



void static inline strided_waffe2_store_svec(float* ptr, waffe2_svec x, const long incx) {
  for (int i=0; i<SIMD_SINGLE_STRIDE;i++) (ptr + i*incx)[0] = x[i];
}

void static inline strided_waffe2_store_dvec(double* ptr, waffe2_dvec x, const long incx) {
  for (int i=0; i<SIMD_DOUBLE_STRIDE;i++) (ptr + i*incx)[0] = x[i];
}

void static inline strided_waffe2_store_i32vec(int32_t* ptr, waffe2_ivec x, const long incx) {
  for (int i=0; i<SIMD_SINGLE_STRIDE;i++) (ptr + i*incx)[0] = x[i];
}

void static inline strided_waffe2_store_u32vec(uint32_t* ptr, waffe2_ivec x, const long incx) {
  for (int i=0; i<SIMD_SINGLE_STRIDE;i++) (ptr + i*incx)[0] = x[i];
}

void static inline strided_waffe2_store_i16vec(int16_t* ptr, waffe2_ivec x, const long incx) {
  for (int i=0; i<SIMD_SINGLE_STRIDE*2;i++) (ptr + i*incx)[0] = x[i];
}

void static inline strided_waffe2_store_u16vec(uint16_t* ptr, waffe2_ivec x, const long incx) {
  for (int i=0; i<SIMD_SINGLE_STRIDE*2;i++) (ptr + i*incx)[0] = x[i];
}

void static inline strided_waffe2_store_i8vec(int8_t* ptr, waffe2_ivec x, const long incx) {
  for (int i=0; i<SIMD_SINGLE_STRIDE*4;i++) (ptr + i*incx)[0] = x[i];
}

void static inline strided_waffe2_store_u8vec(uint8_t* ptr, waffe2_ivec x, const long incx) {
  for (int i=0; i<SIMD_SINGLE_STRIDE*4;i++) (ptr + i*incx)[0] = x[i];
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
  dtype *y_end = y + n * incy;						\
  dtype *y_simd_end = y + (n/stride)*stride;				\
  waffe2_##prefix##vec vx, vy;						\
  if (incx == 1 && incy == 1)						\
    {									\
      while (y != y_simd_end)						\
	{								\
	  vx = waffe2_load_##prefix##vec(x);				\
	  vy = waffe2_load_##prefix##vec(y);				\
	  vy = simd_op_name(vx, vy);					\
	  waffe2_store_##prefix##vec(y, vy);				\
	  x += stride;							\
	  y += stride;							\
	}								\
    }									\
  else if (incx == 1)							\
    {									\
      while (y != y_simd_end)						\
	{								\
	  vx = waffe2_load_##prefix##vec(x);				\
	  vy = make_waffe2_##prefix##vec(y, incy);			\
	  vy = simd_op_name(vx, vy);					\
	  strided_waffe2_store_##prefix##vec(y, vy, incy);	       	\
	  x += stride;							\
	  y += stride;							\
	}								\
    }									\
  else if (incy == 1)							\
    {									\
      while (y != y_simd_end)						\
	{								\
	  vy = waffe2_load_##prefix##vec(y);				\
	  vx = make_waffe2_##prefix##vec(x, incx);			\
	  vy = simd_op_name(vx, vy);					\
	  waffe2_store_##prefix##vec(y, vy);				\
	  x += stride;							\
	  y += stride;							\
	}								\
    }									\
  else									\
    {									\
      while (y != y_simd_end)						\
	{								\
	  vx = make_waffe2_##prefix##vec(x, incx);			\
	  vy = make_waffe2_##prefix##vec(y, incy);			\
	  vy = simd_op_name(vx, vy);					\
	  strided_waffe2_store_##prefix##vec(y, vy, incy);	       	\
	  x += stride;							\
	  y += stride;							\
	}								\
    }									\
  while (y != y_end) {							\
    y[0] = reminder_op_name(x[0], y[0]);				\
    x += incx;								\
    y += incy;								\
  }									\
  };									


#define define_arithmetic_scal_func(define_as, dtype, reminder_op_name) \
  void waffe2_##define_as(const long n, dtype* x, const long incx, dtype* y, const long incy) \
  {									\
    dtype *y_end = y + n * incy;					\
    while (y != y_end) {						\
      (reminder_op_name);						\
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

//                     DEFINE_AS | STRIDE | PREFIX | DTYPE | SIMD_OP | LOOP_REMINDER
// waffe2_sadd ...

define_arithmetic_func(sadd, SIMD_SINGLE_STRIDE, s, float, waffe2_simd_sadd, sas_sadd);
define_arithmetic_func(ssub, SIMD_SINGLE_STRIDE, s, float, waffe2_simd_ssub, sas_ssub);
define_arithmetic_func(smul, SIMD_SINGLE_STRIDE, s, float, waffe2_simd_smul, sas_smul);
define_arithmetic_func(sdiv, SIMD_SINGLE_STRIDE, s, float, waffe2_simd_sdiv, sas_sdiv);

define_arithmetic_func(dadd, SIMD_DOUBLE_STRIDE, d, double, waffe2_simd_dadd, sas_dadd);
define_arithmetic_func(dsub, SIMD_DOUBLE_STRIDE, d, double, waffe2_simd_dsub, sas_dsub);
define_arithmetic_func(dmul, SIMD_DOUBLE_STRIDE, d, double, waffe2_simd_dmul, sas_dmul);
define_arithmetic_func(ddiv, SIMD_DOUBLE_STRIDE, d, double, waffe2_simd_ddiv, sas_ddiv);

define_arithmetic_scal_func(i32add, int32_t, y[0] = x[0] + y[0]);
define_arithmetic_scal_func(i32sub, int32_t, y[0] = x[0] - y[0]);
define_arithmetic_scal_func(i32mul, int32_t, y[0] = x[0] * y[0])
define_arithmetic_scal_func(i32div, int32_t, y[0] = x[0] / y[0]);

define_arithmetic_scal_func(i16add, int16_t, y[0] = x[0] + y[0]);
define_arithmetic_scal_func(i16sub, int16_t, y[0] = x[0] - y[0]);
define_arithmetic_scal_func(i16mul, int16_t, y[0] = x[0] * y[0]);
define_arithmetic_scal_func(i16div, int16_t, y[0] = x[0] / y[0]);

define_arithmetic_scal_func(i8add, int8_t, y[0] = x[0] + y[0]);
define_arithmetic_scal_func(i8sub, int8_t, y[0] = x[0] - y[0]);
define_arithmetic_scal_func(i8mul, int8_t, y[0] = x[0] * y[0]);
define_arithmetic_scal_func(i8div, int8_t, y[0] = x[0] / y[0]);

define_arithmetic_scal_func(u32add, uint32_t, y[0] = x[0] + y[0]);
define_arithmetic_scal_func(u32sub, uint32_t, y[0] = x[0] - y[0]);
define_arithmetic_scal_func(u32mul, uint32_t, y[0] = x[0] * y[0]);
define_arithmetic_scal_func(u32div, uint32_t, y[0] = x[0] / y[0]);

define_arithmetic_scal_func(u16add, uint16_t, y[0] = x[0] + y[0]);
define_arithmetic_scal_func(u16sub, uint16_t, y[0] = x[0] - y[0]);
define_arithmetic_scal_func(u16mul, uint16_t, y[0] = x[0] * y[0]);
define_arithmetic_scal_func(u16div, uint16_t, y[0] = x[0] / y[0]);

define_arithmetic_scal_func(u8add, uint8_t, y[0] = x[0] + y[0]);
define_arithmetic_scal_func(u8sub, uint8_t, y[0] = x[0] - y[0]);
define_arithmetic_scal_func(u8mul, uint8_t, y[0] = x[0] * y[0]);
define_arithmetic_scal_func(u8div, uint8_t, y[0] = x[0] / y[0]);


define_arithmetic_scal_func(i32copy, int32_t, y[0] = x[0]);
define_arithmetic_scal_func(u32copy, uint32_t, y[0] = x[0]);
define_arithmetic_scal_func(i16copy, int16_t, y[0] = x[0]);
define_arithmetic_scal_func(u16copy, uint16_t, y[0] = x[0]);
define_arithmetic_scal_func(i8copy,  int8_t, y[0] = x[0]);
define_arithmetic_scal_func(u8copy,  uint8_t, y[0] = x[0]);

// argmax/argmin
// max/min

// Sets the maximum value in the area to the leading element of y
#define define_maxmin(dpref, prefix, max_or_min, stride, dtype, simd_op, scal_op) \
  void waffe2_##prefix##max_or_min(const long n, dtype* x, const long incx, dtype* y) \
  {									\
    dtype max_value = x[0];						\
    dtype *x_end = x + n * incx;					\
    dtype *x_simd_end = x + (n/stride)*stride;				\
    waffe2_##dpref##vec xv;						\
    waffe2_##dpref##vec maxt = waffe2_load_dscal(max_value);		\
    while (x != x_simd_end)						\
      {									\
	if (incx == 1)							\
	  {								\
	    xv = waffe2_load_##prefix##vec(x);				\
	  }								\
	else								\
	  {								\
	    xv = make_waffe2_##prefix##vec(x, incx);			\
	  }								\
	maxt = simd_op(maxt, xv);					\
	x += stride;							\
      }									\
    dtype candidates[stride];						\
    waffe2_store_##prefix##vec(candidates, maxt);			\
    for (int i=0; i<stride; i++) { max_value = MAX(max_value, candidates[i]); } \
    while (x != x_end) {						\
      max_value = scal_op(max_value, x[0]);				\
      x += incx;							\
    }									\
    y[0] = max_value;							\
  };

define_maxmin(d, d, max, SIMD_DOUBLE_STRIDE, double, waffe2_simd_dmax, MAX);
define_maxmin(d, s, max, SIMD_SINGLE_STRIDE, float,  waffe2_simd_smax, MAX);
define_maxmin(i, i32, max, SIMD_SINGLE_STRIDE, int32_t,  waffe2_simd_i32max, MAX);
define_maxmin(i, u32, max, SIMD_SINGLE_STRIDE, uint32_t,  waffe2_simd_u32max, MAX);
define_maxmin(i, i16, max, SIMD_SINGLE_STRIDE * 2, int16_t,  waffe2_simd_i16max, MAX);
define_maxmin(i, u16, max, SIMD_SINGLE_STRIDE * 2, uint16_t,  waffe2_simd_u16max, MAX);
define_maxmin(i, i8, max, SIMD_SINGLE_STRIDE * 4, int8_t,  waffe2_simd_i8max, MAX);
define_maxmin(i, u8, max, SIMD_SINGLE_STRIDE * 4, uint8_t,  waffe2_simd_u8max, MAX);

define_maxmin(d, d, min, SIMD_DOUBLE_STRIDE, double, waffe2_simd_dmax, MAX);
define_maxmin(d, s, min, SIMD_SINGLE_STRIDE, float,  waffe2_simd_smax, MAX);
define_maxmin(i, i32, min, SIMD_SINGLE_STRIDE, int32_t,  waffe2_simd_i32max, MAX);
define_maxmin(i, u32, min, SIMD_SINGLE_STRIDE, uint32_t,  waffe2_simd_u32max, MAX);
define_maxmin(i, i16, min, SIMD_SINGLE_STRIDE * 2, int16_t,  waffe2_simd_i16max, MAX);
define_maxmin(i, u16, min, SIMD_SINGLE_STRIDE * 2, uint16_t,  waffe2_simd_u16max, MAX);
define_maxmin(i, i8, min, SIMD_SINGLE_STRIDE * 4, int8_t,  waffe2_simd_i8max, MAX);
define_maxmin(i, u8, min, SIMD_SINGLE_STRIDE * 4, uint8_t,  waffe2_simd_u8max, MAX);


// cl-autowrap
// AVX512, SSE2

// SCALAR x TENSOR, INVERSE_TENSOR -> Implemented by Broadcasting
// Data Casting

// TODO: IMPLEMENT SLEEF BACKEND
// Fast Maxmin
// Argmax Argmin
// Where/CompareNodeで関数=#'>等の時、ハードウェアに特化したやつを使う

// ABS
// A>B A<B A<=B A>=B
// A>Scal ...
// max/min with out parameters
// argmax/argmin
// -> SLEEF SIMD MATHEMATICAL
