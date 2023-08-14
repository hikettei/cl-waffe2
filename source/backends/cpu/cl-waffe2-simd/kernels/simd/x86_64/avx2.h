
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
waffe2_ivec static inline waffe2_load_ivec(void* ptr) { return _mm256_loadu_si256((__m256i *)ptr); }

// Setting Scalar
// I'm not enough familar with SIMD developing
// is there anything that broadcast scalar values instead of copying values?
waffe2_svec static inline waffe2_load_sscal(float  scal)  { return _mm256_set1_ps(scal); }
waffe2_dvec static inline waffe2_load_dscal(double scal)  { return _mm256_set1_pd(scal); }

void static inline waffe2_store_svec(float* ptr, waffe2_svec v) { _mm256_storeu_ps(ptr, v); }
void static inline waffe2_store_dvec(double* ptr, waffe2_dvec v){ _mm256_storeu_pd(ptr, v); }
void static inline waffe2_store_ivec(void* ptr, waffe2_ivec v){ _mm256_storeu_si256((__m256i *)ptr, v); }


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

waffe2_svec static inline waffe2_simd_smax(waffe2_svec x, waffe2_svec y) { return _mm256_max_ps(x, y); }
waffe2_dvec static inline waffe2_simd_dmax(waffe2_dvec x, waffe2_dvec y) { return _mm256_max_pd(x, y); }

