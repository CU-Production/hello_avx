//
// Created by Admin on 2023/2/14.
// Floating-Point Arithmetic
//

#include "YmmVal.h"
#include <immintrin.h> // AVX header
#include <iostream>
#include <string>
#define _USE_MATH_DEFINES
#include <math.h>

void PackedMathF32_Iavx(YmmVal c[8], const YmmVal* a, const YmmVal* b);
void PackedMathF64_Iavx(YmmVal c[8], const YmmVal* a, const YmmVal* b);

static void PackedMathF32();
static void PackedMathF64();
static const char* c_OprStr[8] =
        {
                "Add", "Sub", "Mul", "Div", "Min", "Max", "Sqrt a", "Abs b"
        };

int main()
{
    const char nl = '\n';
    PackedMathF32();
    std::cout << nl << std::string(78, '-') << nl;
    PackedMathF64();
}
static void PackedMathF32(void)
{
    YmmVal a, b, c[8];
    const char nl = '\n';
    a.m_F32[0] = 36.0f; b.m_F32[0] = -(float)(1.0 / 9.0);
    a.m_F32[1] = (float)(1.0 / 32.0); b.m_F32[1] = 64.0f;
    a.m_F32[2] = 2.0f; b.m_F32[2] = -0.0625f;
    a.m_F32[3] = 42.0f; b.m_F32[3] = 8.666667f;
    a.m_F32[4] = (float)M_PI; b.m_F32[4] = -4.0f;
    a.m_F32[5] = 18.6f; b.m_F32[5] = -64.0f;
    a.m_F32[6] = 3.0f; b.m_F32[6] = -5.95f;
    a.m_F32[7] = 142.0f; b.m_F32[7] = (float)M_SQRT2;
    PackedMathF32_Iavx(c, &a, &b);
    size_t w = 9;
    std::cout << ("\nResults for PackedMathF32_Iavx\n");
    for (unsigned int i = 0; i < 2; i++)
    {
        std::string s0 = (i == 0) ? "a lo: " : "a hi: ";
        std::string s1 = (i == 0) ? "b lo: " : "b hi: ";
        std::cout << s0 << a.ToStringF32(i) << nl;
        std::cout << s1 << b.ToStringF32(i) << nl;
        for (unsigned int j = 0; j < 8; j++)
        {
            std::cout << std::setw(w) << std::left << c_OprStr[j];
            std::cout << c[j].ToStringF32(i) << nl;
        }
        if (i == 0)
            std::cout << nl;
    }
}

static void PackedMathF64(void)
{
    YmmVal a, b, c[8];
    const char nl = '\n';
    a.m_F64[0] = 2.0; b.m_F64[0] = M_PI;
    a.m_F64[1] = 4.0 ; b.m_F64[1] = M_E;
    a.m_F64[2] = 7.5; b.m_F64[2] = -9.125;
    a.m_F64[3] = 3.0; b.m_F64[3] = -M_PI;
    PackedMathF64_Iavx(c, &a, &b);
    size_t w = 9;
    std::cout << ("\nResults for PackedMathF64_Iavx\n");
    for (unsigned int i = 0; i < 2; i++)
    {
        std::string s0 = (i == 0) ? "a lo: " : "a hi: ";
        std::string s1 = (i == 0) ? "b lo: " : "b hi: ";
        std::cout << s0 << a.ToStringF64(i) << nl;
        std::cout << s1 << b.ToStringF64(i) << nl;
        for (unsigned int j = 0; j < 8; j++)
        {
            std::cout << std::setw(w) << std::left << c_OprStr[j];
            std::cout << c[j].ToStringF64(i) << nl;
        }
        if (i == 0)
            std::cout << nl;
    }
}

void PackedMathF32_Iavx(YmmVal c[8], const YmmVal* a, const YmmVal* b)
{
    __m256 a_vals = _mm256_load_ps((float*)a);
    __m256 b_vals = _mm256_load_ps((float*)b);
    const uint32_t abs_mask_val = 0x7FFFFFFF;
    __m256 abs_mask = _mm256_broadcast_ss((float*)&abs_mask_val);

    __m256 c_vals0 = _mm256_add_ps(a_vals, b_vals);
    __m256 c_vals1 = _mm256_sub_ps(a_vals, b_vals);
    __m256 c_vals2 = _mm256_mul_ps(a_vals, b_vals);
    __m256 c_vals3 = _mm256_div_ps(a_vals, b_vals);
    __m256 c_vals4 = _mm256_min_ps(a_vals, b_vals);
    __m256 c_vals5 = _mm256_max_ps(a_vals, b_vals);
    __m256 c_vals6 = _mm256_sqrt_ps(a_vals);
    __m256 c_vals7 = _mm256_and_ps(b_vals, abs_mask);

    _mm256_store_ps((float*)&c[0], c_vals0);
    _mm256_store_ps((float*)&c[1], c_vals1);
    _mm256_store_ps((float*)&c[2], c_vals2);
    _mm256_store_ps((float*)&c[3], c_vals3);
    _mm256_store_ps((float*)&c[4], c_vals4);
    _mm256_store_ps((float*)&c[5], c_vals5);
    _mm256_store_ps((float*)&c[6], c_vals6);
    _mm256_store_ps((float*)&c[7], c_vals7);
}

void PackedMathF64_Iavx(YmmVal c[8], const YmmVal* a, const YmmVal* b)
{
    __m256d a_vals = _mm256_load_pd((double*)a);
    __m256d b_vals = _mm256_load_pd((double*)b);
    const uint64_t abs_mask_val = 0x7FFFFFFFFFFFFFFF;
    __m256d abs_mask = _mm256_broadcast_sd((double*)&abs_mask_val);

    __m256d c_vals0 = _mm256_add_pd(a_vals, b_vals);
    __m256d c_vals1 = _mm256_sub_pd(a_vals, b_vals);
    __m256d c_vals2 = _mm256_mul_pd(a_vals, b_vals);
    __m256d c_vals3 = _mm256_div_pd(a_vals, b_vals);
    __m256d c_vals4 = _mm256_min_pd(a_vals, b_vals);
    __m256d c_vals5 = _mm256_max_pd(a_vals, b_vals);
    __m256d c_vals6 = _mm256_sqrt_pd(a_vals);
    __m256d c_vals7 = _mm256_and_pd(b_vals, abs_mask);

    _mm256_store_pd((double*)&c[0], c_vals0);
    _mm256_store_pd((double*)&c[1], c_vals1);
    _mm256_store_pd((double*)&c[2], c_vals2);
    _mm256_store_pd((double*)&c[3], c_vals3);
    _mm256_store_pd((double*)&c[4], c_vals4);
    _mm256_store_pd((double*)&c[5], c_vals5);
    _mm256_store_pd((double*)&c[6], c_vals6);
    _mm256_store_pd((double*)&c[7], c_vals7);

}
