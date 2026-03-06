#pragma once
#include <cmath>

namespace mini_aladdin::math {

constexpr double SQRT1_2     = 0.7071067811865475;
constexpr double INV_SQRT2PI = 0.3989422804014327;





#ifdef __CUDACC__
__host__ __device__
#endif
inline double normalCDF(double x) noexcept {
    return 0.5 * erfc(-x * SQRT1_2); // erfc will be from std::erfc and from global ns when compiled with cuda
}

#ifdef __CUDACC__
__host__ __device__
#endif
inline double normalPDF(double x) noexcept {
    return INV_SQRT2PI * exp(-0.5 * x * x); // exp will be from std::exp and from global ns when compiled with cuda
}

} // namespace mini_aladdin::math
