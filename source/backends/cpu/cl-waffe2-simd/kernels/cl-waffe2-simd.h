
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

