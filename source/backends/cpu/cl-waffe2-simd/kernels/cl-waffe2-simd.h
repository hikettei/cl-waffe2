
#define _define_arithmetic_func(define_as, stride, prefix, dtype, simd_op_name, reminder_op_name) \
  void waffe2_##define_as(const long n, dtype* x, const long incx, dtype* y, const long incy);

#define _define_arithmetic_scal_func(define_as, dtype, reminder_op_name) \
  void waffe2_##define_as(const long n, dtype* x, const long incx, dtype* y, const long incy) \


_define_arithmetic_func(sadd, SIMD_SINGLE_STRIDE, s, float, waffe2_simd_sadd, sas_sadd);
_define_arithmetic_func(ssub, SIMD_SINGLE_STRIDE, s, float, waffe2_simd_ssub, sas_ssub);
_define_arithmetic_func(smul, SIMD_SINGLE_STRIDE, s, float, waffe2_simd_smul, sas_smul);
_define_arithmetic_func(sdiv, SIMD_SINGLE_STRIDE, s, float, waffe2_simd_sdiv, sas_sdiv);

_define_arithmetic_func(dadd, SIMD_DOUBLE_STRIDE, d, double, waffe2_simd_dadd, sas_dadd);
_define_arithmetic_func(dsub, SIMD_DOUBLE_STRIDE, d, double, waffe2_simd_dsub, sas_dsub);
_define_arithmetic_func(dmul, SIMD_DOUBLE_STRIDE, d, double, waffe2_simd_dmul, sas_dmul);
_define_arithmetic_func(ddiv, SIMD_DOUBLE_STRIDE, d, double, waffe2_simd_ddiv, sas_ddiv);

_define_arithmetic_scal_func(i32add, int32_t, y[0] = x[0] + y[0]);
_define_arithmetic_scal_func(i32sub, int32_t, y[0] = x[0] - y[0]);
_define_arithmetic_scal_func(i32mul, int32_t, y[0] = x[0] * y[0]);
_define_arithmetic_scal_func(i32div, int32_t, y[0] = x[0] / y[0]);

_define_arithmetic_scal_func(i16add, int16_t, y[0] = x[0] + y[0]);
_define_arithmetic_scal_func(i16sub, int16_t, y[0] = x[0] - y[0]);
_define_arithmetic_scal_func(i16mul, int16_t, y[0] = x[0] * y[0]);
_define_arithmetic_scal_func(i16div, int16_t, y[0] = x[0] / y[0]);

_define_arithmetic_scal_func(i8add, int8_t, y[0] = x[0] + y[0]);
_define_arithmetic_scal_func(i8sub, int8_t, y[0] = x[0] - y[0]);
_define_arithmetic_scal_func(i8mul, int8_t, y[0] = x[0] * y[0]);
_define_arithmetic_scal_func(i8div, int8_t, y[0] = x[0] / y[0]);

_define_arithmetic_scal_func(u32add, uint32_t, y[0] = x[0] + y[0]);
_define_arithmetic_scal_func(u32sub, uint32_t, y[0] = x[0] - y[0]);
_define_arithmetic_scal_func(u32mul, uint32_t, y[0] = x[0] * y[0]);
_define_arithmetic_scal_func(u32div, uint32_t, y[0] = x[0] / y[0]);

_define_arithmetic_scal_func(u16add, uint16_t, y[0] = x[0] + y[0]);
_define_arithmetic_scal_func(u16sub, uint16_t, y[0] = x[0] - y[0]);
_define_arithmetic_scal_func(u16mul, uint16_t, y[0] = x[0] * y[0]);
_define_arithmetic_scal_func(u16div, uint16_t, y[0] = x[0] / y[0]);

_define_arithmetic_scal_func(u8add, uint8_t, y[0] = x[0] + y[0]);
_define_arithmetic_scal_func(u8sub, uint8_t, y[0] = x[0] - y[0]);
_define_arithmetic_scal_func(u8mul, uint8_t, y[0] = x[0] * y[0]);
_define_arithmetic_scal_func(u8div, uint8_t, y[0] = x[0] / y[0]);

// Copying for single/double float -> use BLAS routine
// as for integers -> use functions below

_define_arithmetic_scal_func(i32copy, int32_t, y[0] = x[0]);
_define_arithmetic_scal_func(u32copy, uint32_t, y[0] = x[0]);
_define_arithmetic_scal_func(i16copy, int16_t, y[0] = x[0]);
_define_arithmetic_scal_func(u16copy, uint16_t, y[0] = x[0]);
_define_arithmetic_scal_func(i8copy,  int8_t, y[0] = x[0]);
_define_arithmetic_scal_func(u8copy,  uint8_t, y[0] = x[0]);

#define _define_maxmin(dpref, prefix, max_or_min, stride, dtype, simd_op, scal_op) \
  void waffe2_##prefix##max_or_min(const long n, dtype* x, const long incx, dtype* y); \



_define_maxmin(d, d, max, SIMD_DOUBLE_STRIDE, double, waffe2_simd_dmax, MAX);
_define_maxmin(d, s, max, SIMD_SINGLE_STRIDE, float,  waffe2_simd_smax, MAX);
_define_maxmin(i, i32, max, SIMD_SINGLE_STRIDE, int32_t,  waffe2_simd_i32max, MAX);
_define_maxmin(i, u32, max, SIMD_SINGLE_STRIDE, uint32_t,  waffe2_simd_u32max, MAX);
_define_maxmin(i, i16, max, SIMD_SINGLE_STRIDE * 2, int16_t,  waffe2_simd_i16max, MAX);
_define_maxmin(i, u16, max, SIMD_SINGLE_STRIDE * 2, uint16_t,  waffe2_simd_u16max, MAX);
_define_maxmin(i, i8, max, SIMD_SINGLE_STRIDE * 4, int8_t,  waffe2_simd_i8max, MAX);
_define_maxmin(i, u8, max, SIMD_SINGLE_STRIDE * 4, uint8_t,  waffe2_simd_u8max, MAX);

_define_maxmin(d, d, min, SIMD_DOUBLE_STRIDE, double, waffe2_simd_dmax, MAX);
_define_maxmin(d, s, min, SIMD_SINGLE_STRIDE, float,  waffe2_simd_smax, MAX);
_define_maxmin(i, i32, min, SIMD_SINGLE_STRIDE, int32_t,  waffe2_simd_i32max, MAX);
_define_maxmin(i, u32, min, SIMD_SINGLE_STRIDE, uint32_t,  waffe2_simd_u32max, MAX);
_define_maxmin(i, i16, min, SIMD_SINGLE_STRIDE * 2, int16_t,  waffe2_simd_i16max, MAX);
_define_maxmin(i, u16, min, SIMD_SINGLE_STRIDE * 2, uint16_t,  waffe2_simd_u16max, MAX);
_define_maxmin(i, i8, min, SIMD_SINGLE_STRIDE * 4, int8_t,  waffe2_simd_i8max, MAX);
_define_maxmin(i, u8, min, SIMD_SINGLE_STRIDE * 4, uint8_t,  waffe2_simd_u8max, MAX);


#define _define_cmp(dpref, prefix, op_name, stride, dtype, simd_cmp, rem_cmp) \
  void waffe2_##prefix##op_name(const long n, dtype* x, const long incx, dtype* y, const long incy, dtype* out, const long inco, dtype then_value, dtype else_value) \
  void waffe2_##prefix##op_name##_scal(const long n, dtype* x, const long incx, dtype* out, const long inco, dtype y, dtype then_value, dtype else_value) \
  

_define_cmp(d, d, eq, SIMD_DOUBLE_STRIDE, double, eq, ==);
_define_cmp(s, s, eq, SIMD_SINGLE_STRIDE, float, eq, ==);
_define_cmp(i, i32, eq, SIMD_SINGLE_STRIDE,     int32_t, eq, ==);
_define_cmp(i, i16, eq, SIMD_SINGLE_STRIDE * 2, int16_t, eq, ==);
_define_cmp(i, i8,  eq, SIMD_SINGLE_STRIDE * 4, int8_t,  eq, ==);
_define_cmp(i, u32, eq, SIMD_SINGLE_STRIDE,     uint32_t, eq, ==);
_define_cmp(i, u16, eq, SIMD_SINGLE_STRIDE * 2, uint16_t, eq, ==);
_define_cmp(i, u8,  eq, SIMD_SINGLE_STRIDE * 4, uint8_t,  eq, ==);
    

// A<B, lt
_define_cmp(d, d, lt, SIMD_DOUBLE_STRIDE, double, lt, <);
_define_cmp(s, s, lt, SIMD_SINGLE_STRIDE, float, lt,  <);
_define_cmp(i, i32, lt, SIMD_SINGLE_STRIDE,     int32_t, lt, <);
_define_cmp(i, i16, lt, SIMD_SINGLE_STRIDE * 2, int16_t, lt, <);
_define_cmp(i, i8,  lt, SIMD_SINGLE_STRIDE * 4, int8_t,  lt, <);
_define_cmp(i, u32, lt, SIMD_SINGLE_STRIDE,     uint32_t, lt, <);
_define_cmp(i, u16, lt, SIMD_SINGLE_STRIDE * 2, uint16_t, lt, <);
_define_cmp(i, u8,  lt, SIMD_SINGLE_STRIDE * 4, uint8_t,  lt, <);

// A<=B, le

_define_cmp(d, d, le, SIMD_DOUBLE_STRIDE, double, le, <=);
_define_cmp(s, s, le, SIMD_SINGLE_STRIDE, float, le,  <=);
_define_cmp(i, i32, le, SIMD_SINGLE_STRIDE,     int32_t, le, <=);
_define_cmp(i, i16, le, SIMD_SINGLE_STRIDE * 2, int16_t, le, <=);
_define_cmp(i, i8,  le, SIMD_SINGLE_STRIDE * 4, int8_t,  le, <=);
_define_cmp(i, u32, le, SIMD_SINGLE_STRIDE,     uint32_t, le, <=);
_define_cmp(i, u16, le, SIMD_SINGLE_STRIDE * 2, uint16_t, le, <=);
_define_cmp(i, u8,  le, SIMD_SINGLE_STRIDE * 4, uint8_t,  le, <=);

// A>B gt

_define_cmp(d, d, gt, SIMD_DOUBLE_STRIDE, double, gt, >);
_define_cmp(s, s, gt, SIMD_SINGLE_STRIDE, float, gt,  >);
_define_cmp(i, i32, gt, SIMD_SINGLE_STRIDE,     int32_t, gt, >);
_define_cmp(i, i16, gt, SIMD_SINGLE_STRIDE * 2, int16_t, gt, >);
_define_cmp(i, i8,  gt, SIMD_SINGLE_STRIDE * 4, int8_t,  gt, >);
_define_cmp(i, u32, gt, SIMD_SINGLE_STRIDE,     uint32_t, gt, >);
_define_cmp(i, u16, gt, SIMD_SINGLE_STRIDE * 2, uint16_t, gt, >);
_define_cmp(i, u8,  gt, SIMD_SINGLE_STRIDE * 4, uint8_t,  gt, >);

// A>=B ge

_define_cmp(d, d, ge, SIMD_DOUBLE_STRIDE, double, ge, >=);
_define_cmp(s, s, ge, SIMD_SINGLE_STRIDE, float, ge,  >=);
_define_cmp(i, i32, ge, SIMD_SINGLE_STRIDE,     int32_t, ge, >=);
_define_cmp(i, i16, ge, SIMD_SINGLE_STRIDE * 2, int16_t, ge, >=);
_define_cmp(i, i8,  ge, SIMD_SINGLE_STRIDE * 4, int8_t,  ge, >=);
_define_cmp(i, u32, ge, SIMD_SINGLE_STRIDE,     uint32_t, ge, >=);
_define_cmp(i, u16, ge, SIMD_SINGLE_STRIDE * 2, uint16_t, ge, >=);
_define_cmp(i, u8,  ge, SIMD_SINGLE_STRIDE * 4, uint8_t,  ge, >=);

