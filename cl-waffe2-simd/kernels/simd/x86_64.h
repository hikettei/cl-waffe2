
#include <immintrin.h>

#if defined(__AVX512F__)
    #include "x86_64/avx512f.h"
#elif defined(__AVX2__)
    #include "x86_64/avx2.h"
#else
    #include "x86_64/sse2.h"
#endif

