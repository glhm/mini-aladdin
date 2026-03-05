#pragma once
#include <cmath>

namespace mini_aladdin::math {

constexpr double SQRT1_2 = 0.7071067811865475;
constexpr double INV_SQRT2PI = 0.3989422804014327;

#ifdef __CUDACC__
__host__ __device__
inline double normalCDF(double x) noexcept {
    return 0.5 * erfc(-x * SQRT1_2);
}

__host__ __device__
inline double normalPDF(double x) noexcept {
    return INV_SQRT2PI * exp(-0.5 * x * x);
}
#else
inline double normalCDF(double x) noexcept {
    return 0.5 * std::erfc(-x * SQRT1_2);
}

inline double normalPDF(double x) noexcept {
    return INV_SQRT2PI * std::exp(-0.5 * x * x);
}
#endif

} // namespace mini_aladdin::math
