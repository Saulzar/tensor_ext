#ifndef TH_GENERIC_FILE
#error "You must define TH_GENERIC_FILE before including THGenerateAllTypes.h"
#endif

#define real float
#define Real Cuda
#define TH_REAL_IS_CUDA
#line 1 TH_GENERIC_FILE
#include TH_GENERIC_FILE
#undef real
#undef Real
#undef TH_REAL_IS_CUDA

#undef TH_GENERIC_FILE
