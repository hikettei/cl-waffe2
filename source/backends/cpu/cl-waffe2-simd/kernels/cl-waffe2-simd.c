
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

