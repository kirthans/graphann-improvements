#include "distance.h"
#include <immintrin.h>

float compute_l2sq(const float* a, const float* b, uint32_t dim) {
    uint32_t i = 0;
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    __m256 sum3 = _mm256_setzero_ps();

    // Process 32 elements per iteration (4 registers * 8 floats)
    for (; i + 31 < dim; i += 32) {
        __m256 a0 = _mm256_loadu_ps(a + i);
        __m256 b0 = _mm256_loadu_ps(b + i);
        __m256 diff0 = _mm256_sub_ps(a0, b0);
        sum0 = _mm256_fmadd_ps(diff0, diff0, sum0);

        __m256 a1 = _mm256_loadu_ps(a + i + 8);
        __m256 b1 = _mm256_loadu_ps(b + i + 8);
        __m256 diff1 = _mm256_sub_ps(a1, b1);
        sum1 = _mm256_fmadd_ps(diff1, diff1, sum1);

        __m256 a2 = _mm256_loadu_ps(a + i + 16);
        __m256 b2 = _mm256_loadu_ps(b + i + 16);
        __m256 diff2 = _mm256_sub_ps(a2, b2);
        sum2 = _mm256_fmadd_ps(diff2, diff2, sum2);

        __m256 a3 = _mm256_loadu_ps(a + i + 24);
        __m256 b3 = _mm256_loadu_ps(b + i + 24);
        __m256 diff3 = _mm256_sub_ps(a3, b3);
        sum3 = _mm256_fmadd_ps(diff3, diff3, sum3);
    }

    // Process remaining elements down to multiples of 8
    for (; i + 7 < dim; i += 8) {
        __m256 a0 = _mm256_loadu_ps(a + i);
        __m256 b0 = _mm256_loadu_ps(b + i);
        __m256 diff0 = _mm256_sub_ps(a0, b0);
        sum0 = _mm256_fmadd_ps(diff0, diff0, sum0);
    }

    // Accumulate sum0, sum1, sum2, sum3 into sum0
    sum0 = _mm256_add_ps(sum0, sum1);
    sum2 = _mm256_add_ps(sum2, sum3);
    sum0 = _mm256_add_ps(sum0, sum2);

    // Horizontal sum of the 8 floats in sum0
    // Extract upper 128 bits and add to lower 128
    __m128 sum_half = _mm_add_ps(_mm256_castps256_ps128(sum0), _mm256_extractf128_ps(sum0, 1));
    // Add upper 64 bits to lower 64
    sum_half = _mm_add_ps(sum_half, _mm_movehl_ps(sum_half, sum_half));
    // Add upper 32 bits to lower 32
    sum_half = _mm_add_ss(sum_half, _mm_shuffle_ps(sum_half, sum_half, 0x55));

    float final_sum = _mm_cvtss_f32(sum_half);

    // Scalar fallback for remaining dimensions
    for (; i < dim; ++i) {
        float diff = a[i] - b[i];
        final_sum += diff * diff;
    }

    return final_sum;
}
