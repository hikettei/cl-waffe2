
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

// masks?

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


