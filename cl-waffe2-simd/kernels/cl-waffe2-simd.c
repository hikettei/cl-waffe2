
#pragma SIMD
#pragma GCC optimize ("O3")

// Currently AVX2 is only the supported device
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

#include "im2col.c"

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
  if (incx == 1) {
    return waffe2_load_svec(x);
  } else {
    waffe2_svec out;
    for (int i=0;i<SIMD_SINGLE_STRIDE;i++) out[i] = x[i*incx];
    return out;
  }
}

waffe2_dvec static inline make_waffe2_dvec (double* x, const long incx) {
  if (incx == 1) {
    return waffe2_load_dvec(x);
  } else {
    waffe2_dvec out;
    for (int i=0;i<SIMD_DOUBLE_STRIDE;i++) out[i] = x[i*incx];
    return out;
  }
}

waffe2_ivec static inline make_waffe2_i32vec (int32_t* x, const long incx) {
  if (incx == 1) {
    return waffe2_load_i32vec(x);
  }
  
  waffe2_ivec out;
  for (int i=0;i<SIMD_SINGLE_STRIDE;i++) out[i] = x[i*incx];
  return out;
}

waffe2_ivec static inline make_waffe2_u32vec (uint32_t* x, const long incx) {
  if (incx == 1) {
    return waffe2_load_u32vec(x);
  }
  waffe2_ivec out;
  for (int i=0;i<SIMD_SINGLE_STRIDE;i++) out[i] = x[i*incx];
  return out;
}

waffe2_ivec static inline make_waffe2_i16vec (int16_t* x, const long incx) {
  if (incx == 1) {
    return waffe2_load_i16vec(x);
  }
  waffe2_ivec out;
  for (int i=0;i<SIMD_SINGLE_STRIDE*2;i++) out[i] = x[i*incx];
  return out;
}

waffe2_ivec static inline make_waffe2_u16vec (uint16_t* x, const long incx) {
  if (incx == 1) {
    return waffe2_load_u16vec(x);
  }
  waffe2_ivec out;
  for (int i=0;i<SIMD_SINGLE_STRIDE*2;i++) out[i] = x[i*incx];
  return out;
}

waffe2_ivec static inline make_waffe2_i8vec (int8_t* x, const long incx) {
  if (incx == 1) {
    return waffe2_load_i8vec(x);
  }
  waffe2_ivec out;
  for (int i=0;i<SIMD_SINGLE_STRIDE*4;i++) out[i] = x[i*incx];
  return out;
}

waffe2_ivec static inline make_waffe2_u8vec (uint8_t* x, const long incx) {
  if (incx == 1) {
    return waffe2_load_u8vec(x);
  }
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
// do they works with incN=0?
#define define_arithmetic_func(define_as, stride, prefix, dtype, simd_op_name, reminder_op_name) \
  void waffe2_##define_as(const long n, dtype* x, const long incx, dtype* y, const long incy) \
  {									\
    long count = 0;							\
    const long simd_end_idx = (n/stride)*stride;			\
    waffe2_##prefix##vec vx, vy;					\
    if (incx >= 1) {							\
      if (incx == 1 && incy == 1)					\
	{								\
	  while (count != simd_end_idx)					\
	    {								\
	      vx = waffe2_load_##prefix##vec(x + count);		\
	      vy = waffe2_load_##prefix##vec(y + count);		\
	      vy = simd_op_name(vx, vy);				\
	      waffe2_store_##prefix##vec(x + count, vy);		\
	      count += stride;						\
	    }								\
	}								\
      else if (incx == 1)						\
	{								\
	  while (count != simd_end_idx)				        \
	    {								\
	      vx = waffe2_load_##prefix##vec(x + count);		\
	      vy = make_waffe2_##prefix##vec(y + incy * count, incy);	\
	      vy = simd_op_name(vx, vy);				\
	      waffe2_store_##prefix##vec(x + count, vy);		\
	      count += stride;						\
	    }								\
	}								\
      else if (incy == 1)						\
	{								\
	  while (count != simd_end_idx)					\
	    {								\
	      vy = waffe2_load_##prefix##vec(y + count);		\
	      vx = make_waffe2_##prefix##vec(x + count * incx, incx);	\
	      vy = simd_op_name(vx, vy);				\
	      strided_waffe2_store_##prefix##vec(x + count * incx, vy, incx); \
	      count += stride;						\
	    }								\
	}								\
      else								\
	{								\
	  while (count != simd_end_idx)					\
	    {								\
	      vx = make_waffe2_##prefix##vec(x + count * incx, incx);	\
	      vy = make_waffe2_##prefix##vec(y + count * incy, incy);	\
	      vy = simd_op_name(vx, vy);				\
	      strided_waffe2_store_##prefix##vec(x + count * incx, vy, incx); \
	      count += stride;						\
	    }								\
	}								\
    }									\
    x += count * incx;							\
    y += count * incy;							\
    while (count != n) {						\
      count += 1;							\
      x[0] = reminder_op_name(x[0], y[0]);				\
      x += incx;							\
      y += incy;							\
    }									\
  };									


#define define_arithmetic_scal_func(define_as, dtype, reminder_op_name) \
  void waffe2_##define_as(const long n, dtype* x, const long incx, dtype* y, const long incy) \
  {									\
    long count = 0;							\
    while (count != n) {						\
      count += 1;							\
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

define_arithmetic_scal_func(i32add, int32_t, x[0] = x[0] + y[0]);
define_arithmetic_scal_func(i32sub, int32_t, x[0] = x[0] - y[0]);
define_arithmetic_scal_func(i32mul, int32_t, x[0] = x[0] * y[0])
define_arithmetic_scal_func(i32div, int32_t, x[0] = x[0] / y[0]);

define_arithmetic_scal_func(i16add, int16_t, x[0] = x[0] + y[0]);
define_arithmetic_scal_func(i16sub, int16_t, x[0] = x[0] - y[0]);
define_arithmetic_scal_func(i16mul, int16_t, x[0] = x[0] * y[0]);
define_arithmetic_scal_func(i16div, int16_t, x[0] = x[0] / y[0]);

define_arithmetic_scal_func(i8add, int8_t, x[0] = x[0] + y[0]);
define_arithmetic_scal_func(i8sub, int8_t, x[0] = x[0] - y[0]);
define_arithmetic_scal_func(i8mul, int8_t, x[0] = x[0] * y[0]);
define_arithmetic_scal_func(i8div, int8_t, x[0] = x[0] / y[0]);

define_arithmetic_scal_func(u32add, uint32_t, x[0] = x[0] + y[0]);
define_arithmetic_scal_func(u32sub, uint32_t, x[0] = x[0] - y[0]);
define_arithmetic_scal_func(u32mul, uint32_t, x[0] = x[0] * y[0]);
define_arithmetic_scal_func(u32div, uint32_t, x[0] = x[0] / y[0]);

define_arithmetic_scal_func(u16add, uint16_t, x[0] = x[0] + y[0]);
define_arithmetic_scal_func(u16sub, uint16_t, x[0] = x[0] - y[0]);
define_arithmetic_scal_func(u16mul, uint16_t, x[0] = x[0] * y[0]);
define_arithmetic_scal_func(u16div, uint16_t, x[0] = x[0] / y[0]);

define_arithmetic_scal_func(u8add, uint8_t, x[0] = x[0] + y[0]);
define_arithmetic_scal_func(u8sub, uint8_t, x[0] = x[0] - y[0]);
define_arithmetic_scal_func(u8mul, uint8_t, x[0] = x[0] * y[0]);
define_arithmetic_scal_func(u8div, uint8_t, x[0] = x[0] / y[0]);


define_arithmetic_scal_func(i32copy, int32_t, x[0] = y[0]);
define_arithmetic_scal_func(u32copy, uint32_t, x[0] = y[0]);
define_arithmetic_scal_func(i16copy, int16_t, x[0] = y[0]);
define_arithmetic_scal_func(u16copy, uint16_t, x[0] = y[0]);
define_arithmetic_scal_func(i8copy,  int8_t, x[0] = y[0]);
define_arithmetic_scal_func(u8copy,  uint8_t, x[0] = y[0]);

#define define_inv_func(define_as, dtype, reminder_op_name)		\
  void waffe2_##define_as(const long n, dtype* x, const long incx)	\
  {									\
    long count = 0;							\
    while (count != n) {						\
      count += 1;							\
      (reminder_op_name);						\
      x += incx;							\
    }									\
  };

define_inv_func(dinv, double, x[0] = 1.0/x[0]);
define_inv_func(sinv, float,  x[0] = 1.0/x[0]);

define_inv_func(i32inv, int32_t, x[0] = 1/x[0]);
define_inv_func(i16inv, int16_t, x[0] = 1/x[0]);
define_inv_func(i8inv,  int8_t, x[0] = 1/x[0]);

define_inv_func(u32inv, uint32_t, x[0] = 1/x[0]);
define_inv_func(u16inv, uint16_t, x[0] = 1/x[0]);
define_inv_func(u8inv,  uint8_t, x[0] = 1/x[0]);

// argmax/argmin
// max/min

// Sets the maximum value in the area to the leading element of y
#define define_maxmin(dpref, prefix, max_or_min, stride, dtype, simd_op, scal_op) \
  void waffe2_##prefix##max_or_min(const long n, dtype* x, const long incx, dtype* y) \
  {									\
    dtype max_value = x[0];						\
    long count = 0;							\
    const long simd_end_count =  (n/stride)*stride;			\
    waffe2_##dpref##vec xv;						\
    waffe2_##dpref##vec maxt = waffe2_load_##prefix##scal(max_value);	\
    while (count != simd_end_count)					\
      {									\
	if (incx == 1)							\
	  {								\
	    xv = waffe2_load_##prefix##vec(x + count);	         	\
	  }								\
	else								\
	  {								\
	    xv = make_waffe2_##prefix##vec(x + count*incx, incx);	\
	  }								\
	count += stride;						\
	maxt = simd_op(maxt, xv);					\
      }									\
    dtype candidates[stride];						\
    waffe2_store_##prefix##vec(candidates, maxt);			\
    for (int i=0; i<stride; i++) { max_value = scal_op(max_value, candidates[i]); } \
    x += count * incx;							\
    while (count != n) {						\
      count += 1;							\
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

define_maxmin(d, d, min, SIMD_DOUBLE_STRIDE, double, waffe2_simd_dmin, MIN);
define_maxmin(d, s, min, SIMD_SINGLE_STRIDE, float,  waffe2_simd_smin, MIN);
define_maxmin(i, i32, min, SIMD_SINGLE_STRIDE, int32_t,  waffe2_simd_i32min, MIN);
define_maxmin(i, u32, min, SIMD_SINGLE_STRIDE, uint32_t,  waffe2_simd_u32min, MIN);
define_maxmin(i, i16, min, SIMD_SINGLE_STRIDE * 2, int16_t,  waffe2_simd_i16min, MIN);
define_maxmin(i, u16, min, SIMD_SINGLE_STRIDE * 2, uint16_t,  waffe2_simd_u16min, MIN);
define_maxmin(i, i8, min, SIMD_SINGLE_STRIDE * 4, int8_t,  waffe2_simd_i8min, MIN);
define_maxmin(i, u8, min, SIMD_SINGLE_STRIDE * 4, uint8_t,  waffe2_simd_u8min, MIN);

//
// As for argmin/argmax -> Use Lisp Tensor Implementation due to my limited knowledge :<
// But I've found some projects on SIMD-enabled argmax/argmin impls
//
// https://github.com/minimalrust/argmm
// https://github.com/jvdd/argminmax
//

// out_dtype = uint8.
// Then=1
// Else=0
// inco == 1
// Constraints: out.dtype = uint8, then=1, else=0

// dpref = s d i prefix = i32 u32 s d ...

// TODO: loop iteration could be faster than while iteraton... (cant be said without measuring any benchmarks though...)

#define define_cmp(dpref, prefix, op_name, stride, dtype, simd_cmp, rem_cmp) \
  void waffe2_##prefix##op_name(const long n, dtype* x, const long incx, dtype* y, const long incy, dtype* out, const long inco, dtype then_value, dtype else_value) \
  {                                                                     \
    const long simd_end_count = (n/stride)*stride;			\
    long count = 0;							\
    waffe2_##dpref##vec vx, vy, vo, res;			        \
    waffe2_##dpref##bool mask;						\
    const waffe2_##dpref##vec vthen = waffe2_load_##prefix##scal(then_value); \
    const waffe2_##dpref##vec velse = waffe2_load_##prefix##scal(else_value); \
    if (inco >= 1) {							\
      while (count != simd_end_count)					\
	{								\
	  vx = make_waffe2_##prefix##vec(x + count * incx, incx);	\
	  vy = make_waffe2_##prefix##vec(y + count * incy, incy);	\
	  mask = waffe2_simd_##prefix##simd_cmp(vx, vy);		\
	  res  = waffe2_simd_##prefix##blendv(velse, vthen, mask);      \
	  if (inco == 1) {						\
	    waffe2_store_##prefix##vec(out + count*inco, res);		\
	  } else {							\
	    strided_waffe2_store_##prefix##vec(out + count*inco, res, inco); \
	  }								\
	  count += stride;						\
	}								\
    }									\
    x   += count * incx;						\
    y   += count * incy;						\
    out += count * inco;						\
    while (count != n)							\
      {									\
	if (x[0] rem_cmp y[0]) {				        \
	  out[0] = then_value;						\
	} else {							\
	  out[0] = else_value;						\
	}								\
	count += 1;							\
	x += incx;							\
	y += incy;							\
	out += inco;							\
      }									\
  };

// A = B, eq
define_cmp(d, d, eq, SIMD_DOUBLE_STRIDE, double, eq, ==);
define_cmp(s, s, eq, SIMD_SINGLE_STRIDE, float, eq, ==);
define_cmp(i, i32, eq, SIMD_SINGLE_STRIDE,     int32_t, eq, ==);
define_cmp(i, i16, eq, SIMD_SINGLE_STRIDE * 2, int16_t, eq, ==);
define_cmp(i, i8,  eq, SIMD_SINGLE_STRIDE * 4, int8_t,  eq, ==);
define_cmp(i, u32, eq, SIMD_SINGLE_STRIDE,     uint32_t, eq, ==);
define_cmp(i, u16, eq, SIMD_SINGLE_STRIDE * 2, uint16_t, eq, ==);
define_cmp(i, u8,  eq, SIMD_SINGLE_STRIDE * 4, uint8_t,  eq, ==);
    

// A<B, lt
define_cmp(d, d, lt, SIMD_DOUBLE_STRIDE, double, lt, <);
define_cmp(s, s, lt, SIMD_SINGLE_STRIDE, float, lt,  <);
define_cmp(i, i32, lt, SIMD_SINGLE_STRIDE,     int32_t, lt, <);
define_cmp(i, i16, lt, SIMD_SINGLE_STRIDE * 2, int16_t, lt, <);
define_cmp(i, i8,  lt, SIMD_SINGLE_STRIDE * 4, int8_t,  lt, <);
define_cmp(i, u32, lt, SIMD_SINGLE_STRIDE,     uint32_t, lt, <);
define_cmp(i, u16, lt, SIMD_SINGLE_STRIDE * 2, uint16_t, lt, <);
define_cmp(i, u8,  lt, SIMD_SINGLE_STRIDE * 4, uint8_t,  lt, <);

// A<=B, le

define_cmp(d, d, le, SIMD_DOUBLE_STRIDE, double, le, <=);
define_cmp(s, s, le, SIMD_SINGLE_STRIDE, float, le,  <=);
define_cmp(i, i32, le, SIMD_SINGLE_STRIDE,     int32_t, le, <=);
define_cmp(i, i16, le, SIMD_SINGLE_STRIDE * 2, int16_t, le, <=);
define_cmp(i, i8,  le, SIMD_SINGLE_STRIDE * 4, int8_t,  le, <=);
define_cmp(i, u32, le, SIMD_SINGLE_STRIDE,     uint32_t, le, <=);
define_cmp(i, u16, le, SIMD_SINGLE_STRIDE * 2, uint16_t, le, <=);
define_cmp(i, u8,  le, SIMD_SINGLE_STRIDE * 4, uint8_t,  le, <=);

// A>B gt

define_cmp(d, d, gt, SIMD_DOUBLE_STRIDE, double, gt, >);
define_cmp(s, s, gt, SIMD_SINGLE_STRIDE, float, gt,  >);
define_cmp(i, i32, gt, SIMD_SINGLE_STRIDE,     int32_t, gt, >);
define_cmp(i, i16, gt, SIMD_SINGLE_STRIDE * 2, int16_t, gt, >);
define_cmp(i, i8,  gt, SIMD_SINGLE_STRIDE * 4, int8_t,  gt, >);
define_cmp(i, u32, gt, SIMD_SINGLE_STRIDE,     uint32_t, gt, >);
define_cmp(i, u16, gt, SIMD_SINGLE_STRIDE * 2, uint16_t, gt, >);
define_cmp(i, u8,  gt, SIMD_SINGLE_STRIDE * 4, uint8_t,  gt, >);

// A>=B ge

define_cmp(d, d, ge, SIMD_DOUBLE_STRIDE, double, ge, >=);
define_cmp(s, s, ge, SIMD_SINGLE_STRIDE, float, ge,  >=);
define_cmp(i, i32, ge, SIMD_SINGLE_STRIDE,     int32_t, ge, >=);
define_cmp(i, i16, ge, SIMD_SINGLE_STRIDE * 2, int16_t, ge, >=);
define_cmp(i, i8,  ge, SIMD_SINGLE_STRIDE * 4, int8_t,  ge, >=);
define_cmp(i, u32, ge, SIMD_SINGLE_STRIDE,     uint32_t, ge, >=);
define_cmp(i, u16, ge, SIMD_SINGLE_STRIDE * 2, uint16_t, ge, >=);
define_cmp(i, u8,  ge, SIMD_SINGLE_STRIDE * 4, uint8_t,  ge, >=);


// A>scal etc
// dpref = s d i prefix = i32 u32 s d ...
// waffe2_slt_scal
#define define_scmp(dpref, prefix, op_name, stride, dtype, simd_cmp, rem_cmp) \
  void waffe2_##prefix##op_name##_scal(const long n, dtype* x, const long incx, dtype* out, const long inco, dtype y, dtype then_value, dtype else_value) \
  {                                                                     \
    const long simd_end_idx = (n/stride)*stride;			\
    long count = 0;							\
    waffe2_##dpref##vec vx, vo, res;			                \
    waffe2_##dpref##bool mask;						\
    const waffe2_##dpref##vec vthen = waffe2_load_##prefix##scal(then_value); \
    const waffe2_##dpref##vec velse = waffe2_load_##prefix##scal(else_value); \
    const waffe2_##dpref##vec vy    = waffe2_load_##prefix##scal(y);	\
    if (inco >= 1) {							\
      while (count != simd_end_idx)					\
	{								\
	  vx = make_waffe2_##prefix##vec(x + count*incx, incx);		\
	  vo = make_waffe2_##prefix##vec(out + count*inco, inco);	\
	  mask = waffe2_simd_##prefix##simd_cmp(vx, vy);		\
	  res  = waffe2_simd_##prefix##blendv(velse, vthen, mask);      \
	  if (inco == 1) {						\
	    waffe2_store_##prefix##vec(out + count * inco, res);	\
	  } else {							\
	    strided_waffe2_store_##prefix##vec(out + count * inco, res, inco); \
	  }								\
	  count += stride;						\
	}								\
    }									\
    x += count * incx;							\
    out += count * inco;						\
    while (count != n)							\
      {									\
	if (x[0] rem_cmp y) {				                \
	  out[0] = then_value;						\
	} else {							\
	  out[0] = else_value;						\
	}								\
	count += 1;							\
	x   += incx;							\
	out += inco;							\
      }									\
  };


define_scmp(d, d, eq, SIMD_DOUBLE_STRIDE, double, eq, ==);
define_scmp(s, s, eq, SIMD_SINGLE_STRIDE, float, eq, ==);
define_scmp(i, i32, eq, SIMD_SINGLE_STRIDE,     int32_t, eq, ==);
define_scmp(i, i16, eq, SIMD_SINGLE_STRIDE * 2, int16_t, eq, ==);
define_scmp(i, i8,  eq, SIMD_SINGLE_STRIDE * 4, int8_t,  eq, ==);
define_scmp(i, u32, eq, SIMD_SINGLE_STRIDE,     uint32_t, eq, ==);
define_scmp(i, u16, eq, SIMD_SINGLE_STRIDE * 2, uint16_t, eq, ==);
define_scmp(i, u8,  eq, SIMD_SINGLE_STRIDE * 4, uint8_t,  eq, ==);
    

// A<B, lt
define_scmp(d, d, lt, SIMD_DOUBLE_STRIDE, double, lt, <);
define_scmp(s, s, lt, SIMD_SINGLE_STRIDE, float, lt,  <);
define_scmp(i, i32, lt, SIMD_SINGLE_STRIDE,     int32_t, lt, <);
define_scmp(i, i16, lt, SIMD_SINGLE_STRIDE * 2, int16_t, lt, <);
define_scmp(i, i8,  lt, SIMD_SINGLE_STRIDE * 4, int8_t,  lt, <);
define_scmp(i, u32, lt, SIMD_SINGLE_STRIDE,     uint32_t, lt, <);
define_scmp(i, u16, lt, SIMD_SINGLE_STRIDE * 2, uint16_t, lt, <);
define_scmp(i, u8,  lt, SIMD_SINGLE_STRIDE * 4, uint8_t,  lt, <);

// A<=B, le

define_scmp(d, d, le, SIMD_DOUBLE_STRIDE, double, le, <=);
define_scmp(s, s, le, SIMD_SINGLE_STRIDE, float, le,  <=);
define_scmp(i, i32, le, SIMD_SINGLE_STRIDE,     int32_t, le, <=);
define_scmp(i, i16, le, SIMD_SINGLE_STRIDE * 2, int16_t, le, <=);
define_scmp(i, i8,  le, SIMD_SINGLE_STRIDE * 4, int8_t,  le, <=);
define_scmp(i, u32, le, SIMD_SINGLE_STRIDE,     uint32_t, le, <=);
define_scmp(i, u16, le, SIMD_SINGLE_STRIDE * 2, uint16_t, le, <=);
define_scmp(i, u8,  le, SIMD_SINGLE_STRIDE * 4, uint8_t,  le, <=);

// A>B gt

define_scmp(d, d, gt, SIMD_DOUBLE_STRIDE, double, gt, >);
define_scmp(s, s, gt, SIMD_SINGLE_STRIDE, float, gt,  >);
define_scmp(i, i32, gt, SIMD_SINGLE_STRIDE,     int32_t, gt, >);
define_scmp(i, i16, gt, SIMD_SINGLE_STRIDE * 2, int16_t, gt, >);
define_scmp(i, i8,  gt, SIMD_SINGLE_STRIDE * 4, int8_t,  gt, >);
define_scmp(i, u32, gt, SIMD_SINGLE_STRIDE,     uint32_t, gt, >);
define_scmp(i, u16, gt, SIMD_SINGLE_STRIDE * 2, uint16_t, gt, >);
define_scmp(i, u8,  gt, SIMD_SINGLE_STRIDE * 4, uint8_t,  gt, >);

// A>=B ge

define_scmp(d, d, ge, SIMD_DOUBLE_STRIDE, double, ge, >=);
define_scmp(s, s, ge, SIMD_SINGLE_STRIDE, float, ge,  >=);
define_scmp(i, i32, ge, SIMD_SINGLE_STRIDE,     int32_t, ge, >=);
define_scmp(i, i16, ge, SIMD_SINGLE_STRIDE * 2, int16_t, ge, >=);
define_scmp(i, i8,  ge, SIMD_SINGLE_STRIDE * 4, int8_t,  ge, >=);
define_scmp(i, u32, ge, SIMD_SINGLE_STRIDE,     uint32_t, ge, >=);
define_scmp(i, u16, ge, SIMD_SINGLE_STRIDE * 2, uint16_t, ge, >=);
define_scmp(i, u8,  ge, SIMD_SINGLE_STRIDE * 4, uint8_t,  ge, >=);





// cl-autowrap
// AVX512, SSE2
// Data Casting, SLEEF/SIMD Mathematical Impls, Fast Unfold for Pool/Conv

#define define_math_func(define_as, stride, prefix, dtype, simd_op_name, reminder_op_name) \
  void waffe2_##define_as(const long n, dtype* x, const long incx, dtype* y, const long incy) \
  {									\
    long count = 0;							\
    const long simd_end_idx = (n/stride)*stride;			\
    waffe2_##prefix##vec vx, vy;					\
    if (incx >= 1) {							\
      if (incx == 1 && incy == 1)					\
	{								\
	  while (count != simd_end_idx)					\
	    {								\
	      vx = waffe2_load_##prefix##vec(x + count);		\
	      vy = simd_op_name(vx);			         	\
	      waffe2_store_##prefix##vec(y + count, vy);		\
	      count += stride;						\
	    }								\
	}								\
      else if (incx == 1)						\
	{								\
	  while (count != simd_end_idx)				        \
	    {								\
	      vx = waffe2_load_##prefix##vec(x + count);		\
	      vy = simd_op_name(vx);					\
	      strided_waffe2_store_##prefix##vec(y + count * incy, vy, incy); \
	      count += stride;						\
	    }								\
	}								\
      else if (incy == 1)						\
	{								\
	  while (count != simd_end_idx)					\
	    {								\
	      vx = make_waffe2_##prefix##vec(x + count * incx, incx);	\
	      vy = simd_op_name(vx);				        \
	      waffe2_store_##prefix##vec(y + count, vy);		\
	      count += stride;						\
	    }								\
	}								\
      else								\
	{								\
	  while (count != simd_end_idx)					\
	    {								\
	      vx = make_waffe2_##prefix##vec(x + count * incx, incx);	\
	      vy = simd_op_name(vx);					\
	      strided_waffe2_store_##prefix##vec(y + count * incy, vy, incy); \
	      count += stride;						\
	    }								\
	}								\
    }									\
    x += count * incx;							\
    y += count * incy;							\
    while (count != n) {						\
      count += 1;							\
      y[0] = reminder_op_name(x[0]);					\
      x += incx;							\
      y += incy;							\
    }									\
  };									

define_math_func(dsin, SIMD_DOUBLE_STRIDE, d, double, waffe2_simd_dsin, sin);
define_math_func(ssin, SIMD_SINGLE_STRIDE, s, float,  waffe2_simd_ssin, sin);

define_math_func(dcos, SIMD_DOUBLE_STRIDE, d, double, waffe2_simd_dcos, cos);
define_math_func(scos, SIMD_SINGLE_STRIDE, s, float,  waffe2_simd_scos, cos);

define_math_func(dtan, SIMD_DOUBLE_STRIDE, d, double, waffe2_simd_dtan, tan);
define_math_func(stan, SIMD_SINGLE_STRIDE, s, float,  waffe2_simd_stan, tan);


define_math_func(dasin, SIMD_DOUBLE_STRIDE, d, double, waffe2_simd_dasin, asin);
define_math_func(sasin, SIMD_SINGLE_STRIDE, s, float,  waffe2_simd_sasin, asin);

define_math_func(dacos, SIMD_DOUBLE_STRIDE, d, double, waffe2_simd_dacos, acos);
define_math_func(sacos, SIMD_SINGLE_STRIDE, s, float,  waffe2_simd_sacos, acos);

define_math_func(datan, SIMD_DOUBLE_STRIDE, d, double, waffe2_simd_datan, atan);
define_math_func(satan, SIMD_SINGLE_STRIDE, s, float,  waffe2_simd_satan, atan);

define_math_func(dsinh, SIMD_DOUBLE_STRIDE, d, double, waffe2_simd_dsinh, sinh);
define_math_func(ssinh, SIMD_SINGLE_STRIDE, s, float,  waffe2_simd_ssinh, sinh);

define_math_func(dcosh, SIMD_DOUBLE_STRIDE, d, double, waffe2_simd_dcosh, cosh);
define_math_func(scosh, SIMD_SINGLE_STRIDE, s, float,  waffe2_simd_scosh, cosh);

define_math_func(dtanh, SIMD_DOUBLE_STRIDE, d, double, waffe2_simd_dtanh, tanh);
define_math_func(stanh, SIMD_SINGLE_STRIDE, s, float,  waffe2_simd_stanh, tanh);

define_math_func(dasinh, SIMD_DOUBLE_STRIDE, d, double, waffe2_simd_dsinh, asinh);
define_math_func(sasinh, SIMD_SINGLE_STRIDE, s, float,  waffe2_simd_ssinh, asinh);

define_math_func(dacosh, SIMD_DOUBLE_STRIDE, d, double, waffe2_simd_dcosh, acosh);
define_math_func(sacosh, SIMD_SINGLE_STRIDE, s, float,  waffe2_simd_scosh, acosh);

define_math_func(datanh, SIMD_DOUBLE_STRIDE, d, double, waffe2_simd_dtanh, atanh);
define_math_func(satanh, SIMD_SINGLE_STRIDE, s, float,  waffe2_simd_stanh, atanh);


// loge log(1+x) log2 log10

define_math_func(dlog, SIMD_DOUBLE_STRIDE, d, double, waffe2_simd_dlog, log);
define_math_func(slog, SIMD_SINGLE_STRIDE, s, float,  waffe2_simd_slog, log);

define_math_func(dlog1p, SIMD_DOUBLE_STRIDE, d, double, waffe2_simd_dlog1p, log1p);
define_math_func(slog1p, SIMD_SINGLE_STRIDE, s, float,  waffe2_simd_slog1p, log1p);

define_math_func(dlog2, SIMD_DOUBLE_STRIDE, d, double, waffe2_simd_dlog2, log2);
define_math_func(slog2, SIMD_SINGLE_STRIDE, s, float,  waffe2_simd_slog2, log2);

define_math_func(dlog10, SIMD_DOUBLE_STRIDE, d, double, waffe2_simd_dlog10, log10);
define_math_func(slog10, SIMD_SINGLE_STRIDE, s, float,  waffe2_simd_slog10, log10);

define_math_func(dexp, SIMD_DOUBLE_STRIDE, d, double, waffe2_simd_dexp, exp);
define_math_func(sexp, SIMD_SINGLE_STRIDE, s, float,  waffe2_simd_sexp, exp);

define_math_func(dabs, SIMD_DOUBLE_STRIDE, d, double, waffe2_simd_dabs, fabs);
define_math_func(sabs, SIMD_SINGLE_STRIDE, s, float,  waffe2_simd_sabs, fabs);

define_math_func(dsqrt, SIMD_DOUBLE_STRIDE, d, double, waffe2_simd_dsqrt, sqrt);
define_math_func(ssqrt, SIMD_SINGLE_STRIDE, s, float,  waffe2_simd_ssqrt, sqrt);

define_math_func(dcbrt, SIMD_DOUBLE_STRIDE, d, double, waffe2_simd_dcbrt, cbrt);
define_math_func(scbrt, SIMD_SINGLE_STRIDE, s, float,  waffe2_simd_scbrt, cbrt);

#define define_math_func_pow(define_as, stride, prefix, dtype, simd_op_name, reminder_op_name) \
  void waffe2_##define_as(const long n, dtype* x, const long incx, dtype pow_n, dtype* y, const long incy) \
  {									\
    long count = 0;							\
    const long simd_end_idx = (n/stride)*stride;			\
    waffe2_##prefix##vec vx, vy;					\
    waffe2_##prefix##vec pn = waffe2_load_##prefix##scal(pow_n);        \
    if (incx >= 1) {							\
      if (incx == 1 && incy == 1)					\
	{								\
	  while (count != simd_end_idx)					\
	    {								\
	      vx = waffe2_load_##prefix##vec(x + count);		\
	      vy = simd_op_name(vx, pn);			        \
	      waffe2_store_##prefix##vec(y + count, vy);		\
	      count += stride;						\
	    }								\
	}								\
      else if (incx == 1)						\
	{								\
	  while (count != simd_end_idx)				        \
	    {								\
	      vx = waffe2_load_##prefix##vec(x + count);		\
	      vy = simd_op_name(vx, pn);				\
	      strided_waffe2_store_##prefix##vec(y + count * incy, vy, incy); \
	      count += stride;						\
	    }								\
	}								\
      else if (incy == 1)						\
	{								\
	  while (count != simd_end_idx)					\
	    {								\
	      vx = make_waffe2_##prefix##vec(x + count * incx, incx);	\
	      vy = simd_op_name(vx, pn);				\
	      waffe2_store_##prefix##vec(y + count, vy);		\
	      count += stride;						\
	    }								\
	}								\
      else								\
	{								\
	  while (count != simd_end_idx)					\
	    {								\
	      vx = make_waffe2_##prefix##vec(x + count * incx, incx);	\
	      vy = simd_op_name(vx, pn);				\
	      strided_waffe2_store_##prefix##vec(y + count * incy, vy, incy); \
	      count += stride;						\
	    }								\
	}								\
    }									\
    x += count * incx;							\
    y += count * incy;							\
    while (count != n) {						\
      count += 1;							\
      y[0] = reminder_op_name(x[0], pow_n);				\
      x += incx;							\
      y += incy;							\
    }									\
  };									

define_math_func_pow(dpow, SIMD_DOUBLE_STRIDE, d, double, waffe2_simd_dpow, pow);
define_math_func_pow(spow, SIMD_SINGLE_STRIDE, s, float,  waffe2_simd_spow, pow);

