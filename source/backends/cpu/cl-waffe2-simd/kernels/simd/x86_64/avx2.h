
/*
 Workload
 
single, double-float supporting for AVX2 (WIP)
sparse matrix supporting
  ...
*/

// single-float
typedef __m256 waffe2_svec;

// double-float
typedef __m256d waffe2_dvec;

// integers
typedef __m256i waffe2_ivec;

// masks
typedef __m256  waffe2_sbool;
typedef __m256d waffe2_dbool;
typedef __m256i waffe2_ibool;

#define SIMD_SINGLE_STRIDE 8
#define SIMD_DOUBLE_STRIDE 4

// (AVX, AVX2 2008 onwards)
// __mm256  <- float32 * 8
// __mm256i <- 4*uint64, 8*uint32, 16*uint16, 32*uint8
// __mm256d <- 4*float64

// Storing
waffe2_svec static inline waffe2_load_svec(float* ptr)  { return _mm256_loadu_ps(ptr); }
waffe2_dvec static inline waffe2_load_dvec(double* ptr) { return _mm256_loadu_pd(ptr); }

waffe2_ivec static inline waffe2_load_i32vec(int32_t* ptr) { return _mm256_loadu_si256((__m256i *)ptr); }
waffe2_ivec static inline waffe2_load_i16vec(int16_t* ptr) { return _mm256_loadu_si256((__m256i *)ptr); }
waffe2_ivec static inline waffe2_load_i8vec (int8_t* ptr) { return _mm256_loadu_si256((__m256i *)ptr); }

waffe2_ivec static inline waffe2_load_u32vec(uint32_t* ptr) { return _mm256_loadu_si256((__m256i *)ptr); }
waffe2_ivec static inline waffe2_load_u16vec(uint16_t* ptr) { return _mm256_loadu_si256((__m256i *)ptr); }
waffe2_ivec static inline waffe2_load_u8vec (uint8_t* ptr) { return _mm256_loadu_si256((__m256i *)ptr); }


// Setting Scalar
// I'm not enough familar with SIMD developing
// is there anything that broadcast scalar values instead of copying values?
// Scalar -> Register

waffe2_svec static inline waffe2_load_sscal(float  scal)  { return _mm256_set1_ps(scal); }
waffe2_dvec static inline waffe2_load_dscal(double scal)  { return _mm256_set1_pd(scal); }

waffe2_dvec static inline waffe2_load_i32scal(int32_t scal)  { return _mm256_set1_epi32(scal); }
waffe2_dvec static inline waffe2_load_i16scal(int16_t scal)  { return _mm256_set1_epi16(scal); }
waffe2_dvec static inline waffe2_load_i8scal(int8_t scal)    { return _mm256_set1_epi8(scal); }

waffe2_dvec static inline waffe2_load_u32scal(uint32_t scal)  { return _mm256_set1_epi32(scal); }
waffe2_dvec static inline waffe2_load_u16scal(uint16_t scal)  { return _mm256_set1_epi16(scal); }
waffe2_dvec static inline waffe2_load_u8scal(uint8_t scal)    { return _mm256_set1_epi8(scal); }

// loading values
void static inline waffe2_store_svec(float* ptr, waffe2_svec v) { _mm256_storeu_ps(ptr, v); }
void static inline waffe2_store_dvec(double* ptr, waffe2_dvec v){ _mm256_storeu_pd(ptr, v); }

void static inline waffe2_store_i32vec(int32_t* ptr, waffe2_ivec v){ _mm256_storeu_si256((__m256i *)ptr, v); }
void static inline waffe2_store_i16vec(int16_t* ptr, waffe2_ivec v){ _mm256_storeu_si256((__m256i *)ptr, v); }
void static inline waffe2_store_i8vec (int8_t* ptr, waffe2_ivec v){ _mm256_storeu_si256((__m256i *)ptr, v); }

void static inline waffe2_store_u32vec(uint32_t* ptr, waffe2_ivec v){ _mm256_storeu_si256((__m256i *)ptr, v); }
void static inline waffe2_store_u16vec(uint16_t* ptr, waffe2_ivec v){ _mm256_storeu_si256((__m256i *)ptr, v); }
void static inline waffe2_store_u8vec (uint8_t* ptr, waffe2_ivec v){  _mm256_storeu_si256((__m256i *)ptr, v); }


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Arithmetic
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

waffe2_svec static inline waffe2_simd_sadd(waffe2_svec x, waffe2_svec y) { return _mm256_add_ps(x, y); }
waffe2_svec static inline waffe2_simd_ssub(waffe2_svec x, waffe2_svec y) { return _mm256_sub_ps(x, y); }
waffe2_svec static inline waffe2_simd_smul(waffe2_svec x, waffe2_svec y) { return _mm256_mul_ps(x, y); }
waffe2_svec static inline waffe2_simd_sdiv(waffe2_svec x, waffe2_svec y) { return _mm256_div_ps(x, y); }


waffe2_dvec static inline waffe2_simd_dadd(waffe2_dvec x, waffe2_dvec y) { return _mm256_add_pd(x, y); }
waffe2_dvec static inline waffe2_simd_dsub(waffe2_dvec x, waffe2_dvec y) { return _mm256_sub_pd(x, y); }
waffe2_dvec static inline waffe2_simd_dmul(waffe2_dvec x, waffe2_dvec y) { return _mm256_mul_pd(x, y); }
waffe2_dvec static inline waffe2_simd_ddiv(waffe2_dvec x, waffe2_dvec y) { return _mm256_div_pd(x, y); }

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// max/min
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

waffe2_svec static inline waffe2_simd_smax(waffe2_svec x, waffe2_svec y) { return _mm256_max_ps(x, y); }
waffe2_dvec static inline waffe2_simd_dmax(waffe2_dvec x, waffe2_dvec y) { return _mm256_max_pd(x, y); }

waffe2_ivec static inline waffe2_simd_i32max(waffe2_ivec x, waffe2_ivec y) { return _mm256_max_epi32(x, y); }
waffe2_ivec static inline waffe2_simd_i16max(waffe2_ivec x, waffe2_ivec y) { return _mm256_max_epi16(x, y); }
waffe2_ivec static inline waffe2_simd_i8max(waffe2_ivec x, waffe2_ivec y)  { return _mm256_max_epi8(x, y); }

waffe2_ivec static inline waffe2_simd_u32max(waffe2_ivec x, waffe2_ivec y) { return _mm256_max_epu32(x, y); }
waffe2_ivec static inline waffe2_simd_u16max(waffe2_ivec x, waffe2_ivec y) { return _mm256_max_epu16(x, y); }
waffe2_ivec static inline waffe2_simd_u8max (waffe2_ivec x, waffe2_ivec y) { return _mm256_max_epu8(x, y); }


waffe2_svec static inline waffe2_simd_smin(waffe2_svec x, waffe2_svec y) { return _mm256_min_ps(x, y); }
waffe2_dvec static inline waffe2_simd_dmin(waffe2_dvec x, waffe2_dvec y) { return _mm256_min_pd(x, y); }

waffe2_ivec static inline waffe2_simd_i32min(waffe2_ivec x, waffe2_ivec y) { return _mm256_min_epi32(x, y); }
waffe2_ivec static inline waffe2_simd_i16min(waffe2_ivec x, waffe2_ivec y) { return _mm256_min_epi16(x, y); }
waffe2_ivec static inline waffe2_simd_i8min(waffe2_ivec x, waffe2_ivec y)  { return _mm256_min_epi8(x, y); }

waffe2_ivec static inline waffe2_simd_u32min(waffe2_ivec x, waffe2_ivec y) { return _mm256_min_epu32(x, y); }
waffe2_ivec static inline waffe2_simd_u16min(waffe2_ivec x, waffe2_ivec y) { return _mm256_min_epu16(x, y); }
waffe2_ivec static inline waffe2_simd_u8min (waffe2_ivec x, waffe2_ivec y) { return _mm256_min_epu8(x, y); }


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//  Compares
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Referenced: https://qiita.com/fukushima1981/items/5001079900b328696859
// Options:
/* 0: OP := _CMP_EQ_OQ */
/* 1: OP := _CMP_LT_OS */
/* 2: OP := _CMP_LE_OS */
/* 3: OP := _CMP_UNORD_Q */
/* 4: OP := _CMP_NEQ_UQ */
/* 5: OP := _CMP_NLT_US */
/* 6: OP := _CMP_NLE_US */
/* 7: OP := _CMP_ORD_Q */
/* 8: OP := _CMP_EQ_UQ */
/* 9: OP := _CMP_NGE_US */
/* 10: OP := _CMP_NGT_US */
/* 11: OP := _CMP_FALSE_OQ */
/* 12: OP := _CMP_NEQ_OQ */
/* 13: OP := _CMP_GE_OS */
/* 14: OP := _CMP_GT_OS */
/* 15: OP := _CMP_TRUE_UQ */
/* 16: OP := _CMP_EQ_OS */
/* 17: OP := _CMP_LT_OQ */
/* 18: OP := _CMP_LE_OQ */
/* 19: OP := _CMP_UNORD_S */
/* 20: OP := _CMP_NEQ_US */
/* 21: OP := _CMP_NLT_UQ */
/* 22: OP := _CMP_NLE_UQ */
/* 23: OP := _CMP_ORD_S */
/* 24: OP := _CMP_EQ_US */
/* 25: OP := _CMP_NGE_UQ */
/* 26: OP := _CMP_NGT_UQ */
/* 27: OP := _CMP_FALSE_OS */
/* 28: OP := _CMP_NEQ_OS */
/* 29: OP := _CMP_GE_OQ */
/* 30: OP := _CMP_GT_OQ */
/* 31: OP := _CMP_TRUE_US */


// not

waffe2_ivec static inline waffe2_simd_i8not (waffe2_ivec src) {
  return _mm256_xor_si256(src, _mm256_cmpeq_epi8(src, src));
}

// A=B
waffe2_dbool static inline waffe2_simd_deq(waffe2_dvec x, waffe2_dvec y) { return _mm256_cmp_pd(x, y, 0); }
waffe2_sbool static inline waffe2_simd_seq(waffe2_svec x, waffe2_svec y) { return _mm256_cmp_ps(x, y, 0); }

waffe2_ivec  static inline waffe2_simd_i32eq(waffe2_ivec x, waffe2_ivec y) { return _mm256_cmpeq_epi32(x, y); }
waffe2_ivec  static inline waffe2_simd_i16eq(waffe2_ivec x, waffe2_ivec y) { return _mm256_cmpeq_epi16(x, y); }
waffe2_ivec  static inline waffe2_simd_i8eq(waffe2_ivec x, waffe2_ivec y) { return _mm256_cmpeq_epi8(x, y); }

waffe2_ivec  static inline waffe2_simd_u32eq(waffe2_ivec x, waffe2_ivec y) { return _mm256_cmpeq_epi32(x, y); }
waffe2_ivec  static inline waffe2_simd_u16eq(waffe2_ivec x, waffe2_ivec y) { return _mm256_cmpeq_epi16(x, y); }
waffe2_ivec  static inline waffe2_simd_u8eq(waffe2_ivec x, waffe2_ivec y)  { return _mm256_cmpeq_epi8(x, y); }

// A<B lt _CMP_LT_OQ

waffe2_dbool static inline waffe2_simd_dlt(waffe2_dvec x, waffe2_dvec y) { return _mm256_cmp_pd(x, y, 17); }
waffe2_sbool static inline waffe2_simd_slt(waffe2_svec x, waffe2_svec y) { return _mm256_cmp_ps(x, y, 17); }

waffe2_ivec  static inline waffe2_simd_i32lt(waffe2_ivec x, waffe2_ivec y) { return _mm256_cmpgt_epi32(y, x); }
waffe2_ivec  static inline waffe2_simd_i16lt(waffe2_ivec x, waffe2_ivec y) { return _mm256_cmpgt_epi16(y, x); }
waffe2_ivec  static inline waffe2_simd_i8lt(waffe2_ivec x, waffe2_ivec y) { return _mm256_cmpgt_epi8(y, x); }

waffe2_ivec  static inline waffe2_simd_u32lt(waffe2_ivec x, waffe2_ivec y) { return _mm256_cmpgt_epi32(y, x); }
waffe2_ivec  static inline waffe2_simd_u16lt(waffe2_ivec x, waffe2_ivec y) { return _mm256_cmpgt_epi16(y, x); }
waffe2_ivec  static inline waffe2_simd_u8lt(waffe2_ivec x, waffe2_ivec y)  { return _mm256_cmpgt_epi8(y, x); }

// A>B, greater than, _CMP_GT_OQ=30

waffe2_dbool static inline waffe2_simd_dgt(waffe2_dvec x, waffe2_dvec y) { return _mm256_cmp_pd(x, y, 30); }
waffe2_sbool static inline waffe2_simd_sgt(waffe2_svec x, waffe2_svec y) { return _mm256_cmp_ps(x, y, 30); }

waffe2_ivec  static inline waffe2_simd_i32gt(waffe2_ivec x, waffe2_ivec y) { return _mm256_cmpgt_epi32(x, y); }
waffe2_ivec  static inline waffe2_simd_i16gt(waffe2_ivec x, waffe2_ivec y) { return _mm256_cmpgt_epi16(x, y); }
waffe2_ivec  static inline waffe2_simd_i8gt(waffe2_ivec x, waffe2_ivec y) { return _mm256_cmpgt_epi8(x, y); }

waffe2_ivec  static inline waffe2_simd_u32gt(waffe2_ivec x, waffe2_ivec y) { return _mm256_cmpgt_epi32(x, y); }
waffe2_ivec  static inline waffe2_simd_u16gt(waffe2_ivec x, waffe2_ivec y) { return _mm256_cmpgt_epi16(x, y); }
waffe2_ivec  static inline waffe2_simd_u8gt(waffe2_ivec x, waffe2_ivec y)  { return _mm256_cmpgt_epi8(x, y); }


// A<=B, LE _CMP_LE_OQ=18
waffe2_dbool static inline waffe2_simd_dle(waffe2_dvec x, waffe2_dvec y) { return _mm256_cmp_pd(x, y, 18); }
waffe2_sbool static inline waffe2_simd_sle(waffe2_svec x, waffe2_svec y) { return _mm256_cmp_ps(x, y, 18); }

waffe2_ivec static inline waffe2_simd_i32le(waffe2_ivec x, waffe2_ivec y) { return waffe2_simd_i8not(waffe2_simd_i32gt(x, y)); }
waffe2_ivec static inline waffe2_simd_i16le(waffe2_ivec x, waffe2_ivec y) { return waffe2_simd_i8not(waffe2_simd_i16gt(x, y)); }
waffe2_ivec static inline waffe2_simd_i8le(waffe2_ivec x, waffe2_ivec y)  { return waffe2_simd_i8not(waffe2_simd_i8gt(x, y)); }

waffe2_ivec static inline waffe2_simd_u32le(waffe2_ivec x, waffe2_ivec y) { return waffe2_simd_i8not(waffe2_simd_u32gt(x, y)); }
waffe2_ivec static inline waffe2_simd_u16le(waffe2_ivec x, waffe2_ivec y) { return waffe2_simd_i8not(waffe2_simd_u16gt(x, y)); }
waffe2_ivec static inline waffe2_simd_u8le(waffe2_ivec x, waffe2_ivec y)  { return waffe2_simd_i8not(waffe2_simd_u8gt(x, y)); }

// A>=B, greater equal, _CMP_GE_OQ=29
waffe2_dbool static inline waffe2_simd_dge(waffe2_dvec x, waffe2_dvec y) { return _mm256_cmp_pd(x, y, 29); }
waffe2_sbool static inline waffe2_simd_sge(waffe2_svec x, waffe2_svec y) { return _mm256_cmp_ps(x, y, 29); }


waffe2_ivec static inline waffe2_simd_i32ge(waffe2_ivec x, waffe2_ivec y) { return waffe2_simd_i8not(waffe2_simd_i32gt(y, x)); }
waffe2_ivec static inline waffe2_simd_i16ge(waffe2_ivec x, waffe2_ivec y) { return waffe2_simd_i8not(waffe2_simd_i16gt(y, x)); }
waffe2_ivec static inline waffe2_simd_i8ge(waffe2_ivec x, waffe2_ivec y)  { return waffe2_simd_i8not(waffe2_simd_i8gt(y, x)); }

waffe2_ivec static inline waffe2_simd_u32ge(waffe2_ivec x, waffe2_ivec y) { return waffe2_simd_i8not(waffe2_simd_u32gt(y, x)); }
waffe2_ivec static inline waffe2_simd_u16ge(waffe2_ivec x, waffe2_ivec y) { return waffe2_simd_i8not(waffe2_simd_u16gt(y, x)); }
waffe2_ivec static inline waffe2_simd_u8ge(waffe2_ivec x, waffe2_ivec y)  { return waffe2_simd_i8not(waffe2_simd_u8gt(y, x));  }


// ~~~~~~~~~~~~~~~~~~
// Blend
// ~~~~~~~~~~~~~~~~~~

waffe2_dvec static inline waffe2_simd_dblendv(waffe2_dvec x, waffe2_dvec y, waffe2_dbool mask) { return _mm256_blendv_pd(x, y, mask); }
waffe2_svec static inline waffe2_simd_sblendv(waffe2_svec x, waffe2_svec y, waffe2_sbool mask) { return _mm256_blendv_ps(x, y, mask); }

waffe2_ivec static inline waffe2_simd_i32blendv(waffe2_ivec x, waffe2_ivec y, waffe2_ivec mask) { return _mm256_blendv_epi8(x, y, mask); }
waffe2_ivec static inline waffe2_simd_i16blendv(waffe2_ivec x, waffe2_ivec y, waffe2_ivec mask) { return _mm256_blendv_epi8(x, y, mask); }
waffe2_ivec static inline waffe2_simd_i8blendv(waffe2_ivec x, waffe2_ivec y, waffe2_ivec mask) { return _mm256_blendv_epi8(x, y, mask); }

waffe2_ivec static inline waffe2_simd_u32blendv(waffe2_ivec x, waffe2_ivec y, waffe2_ivec mask) { return _mm256_blendv_epi8(x, y, mask); }
waffe2_ivec static inline waffe2_simd_u16blendv(waffe2_ivec x, waffe2_ivec y, waffe2_ivec mask) { return _mm256_blendv_epi8(x, y, mask); }
waffe2_ivec static inline waffe2_simd_u8blendv(waffe2_ivec x, waffe2_ivec y, waffe2_ivec mask) { return _mm256_blendv_epi8(x, y, mask); }

