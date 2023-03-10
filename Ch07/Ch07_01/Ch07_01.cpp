//
// Created by Admin on 2023/2/14.
// AVX512 Integer Arithmetic
//

#include "ZmmVal.h"
#include <immintrin.h> // AVX header
#include <zmmintrin.h> // AVX512 header
#include <iostream>

/**
 * <mmintrin.h>  MMX
 * <xmmintrin.h> SSE
 * <emmintrin.h> SSE2
 * <pmmintrin.h> SSE3
 * <tmmintrin.h> SSSE3
 * <smmintrin.h> SSE4.1
 * <nmmintrin.h> SSE4.2
 * <ammintrin.h> SSE4A
 * <wmmintrin.h> AES
 * <immintrin.h> AVX, AVX2, FMA
 */

void MathI16_Iavx512(ZmmVal c[6], const ZmmVal* a, const ZmmVal* b);
void MathI64_Iavx512(ZmmVal c[6], const ZmmVal* a, const ZmmVal* b);

static void MathI16(void);
static void MathI64(void);

int main()
{
    MathI16();
    MathI64();
    return 0;
}
static void MathI16(void)
{
    ZmmVal a, b, c[6];

    a.m_I16[0] = 10; b.m_I16[0] = 100;
    a.m_I16[1] = 20; b.m_I16[1] = 200;
    a.m_I16[2] = 30; b.m_I16[2] = 300;
    a.m_I16[3] = 40; b.m_I16[3] = 400;
    a.m_I16[4] = 50; b.m_I16[4] = 500;
    a.m_I16[5] = 60; b.m_I16[5] = 600;
    a.m_I16[6] = 70; b.m_I16[6] = 700;
    a.m_I16[7] = 80; b.m_I16[7] = 800;

    a.m_I16[8] = 1000; b.m_I16[8] = -100;
    a.m_I16[9] = 2000; b.m_I16[9] = 200;
    a.m_I16[10] = 3000; b.m_I16[10] = -300;
    a.m_I16[11] = 4000; b.m_I16[11] = 400;
    a.m_I16[12] = 5000; b.m_I16[12] = -500;
    a.m_I16[13] = 6000; b.m_I16[13] = 600;
    a.m_I16[14] = 7000; b.m_I16[14] = -700;
    a.m_I16[15] = 8000; b.m_I16[15] = 800;

    a.m_I16[16] = -1000; b.m_I16[16] = 100;
    a.m_I16[17] = -2000; b.m_I16[17] = -200;
    a.m_I16[18] = 3000; b.m_I16[18] = 303;
    a.m_I16[19] = 4000; b.m_I16[19] = -400;
    a.m_I16[20] = -5000; b.m_I16[20] = 500;
    a.m_I16[21] = -6000; b.m_I16[21] = -600;
    a.m_I16[22] = -7000; b.m_I16[22] = 700;
    a.m_I16[23] = -8000; b.m_I16[23] = 800;

    a.m_I16[24] = 30000; b.m_I16[24] = 3000; // add overflow
    a.m_I16[25] = 6000; b.m_I16[25] = 32000; // add overflow
    a.m_I16[26] = -25000; b.m_I16[26] = -27000; // add overflow
    a.m_I16[27] = 8000; b.m_I16[27] = 28700; // add overflow
    a.m_I16[28] = 2000; b.m_I16[28] = -31000; // sub overflow
    a.m_I16[29] = 4000; b.m_I16[29] = -30000; // sub overflow
    a.m_I16[30] = -3000; b.m_I16[30] = 32000; // sub overflow
    a.m_I16[31] = -15000; b.m_I16[31] = 24000; // sub overflow
    MathI16_Iavx512(c, &a, &b);

    std::cout <<"\nResults for MathI16\n\n";
    std::cout << " i a b add adds sub subs min max\n";
    std::cout << "------------------------------------------------------------------\n";
    for (size_t i = 0; i < 32; i++)
    {
        std::cout << std::setw(2) << i << ' ';
        std::cout << std::setw(8) << a.m_I16[i] << ' ';
        std::cout << std::setw(8) << b.m_I16[i] << ' ';
        std::cout << std::setw(8) << c[0].m_I16[i] << ' ';
        std::cout << std::setw(8) << c[1].m_I16[i] << ' ';
        std::cout << std::setw(8) << c[2].m_I16[i] << ' ';
        std::cout << std::setw(8) << c[3].m_I16[i] << ' ';
        std::cout << std::setw(8) << c[4].m_I16[i] << ' ';
        std::cout << std::setw(8) << c[5].m_I16[i] << '\n';
    }
}
static void MathI64(void)
{
    ZmmVal a, b, c[6];
    a.m_I64[0] = 64; b.m_I64[0] = 4;
    a.m_I64[1] = 1024; b.m_I64[1] = 5;
    a.m_I64[2] = -2048; b.m_I64[2] = 2;
    a.m_I64[3] = 8192; b.m_I64[3] = 5;
    a.m_I64[4] = -256; b.m_I64[4] = 8;
    a.m_I64[5] = 4096; b.m_I64[5] = 7;
    a.m_I64[6] = 16; b.m_I64[6] = 3;
    a.m_I64[7] = 512; b.m_I64[7] = 6;
    MathI64_Iavx512(c, &a, &b);
    std::cout << "\nResults for MathI64\n\n";
    std::cout << " i a b add sub mul sll sra abs\n";
    std::cout << "----------------------------------------------------------------------\n";
    for (size_t i = 0; i < 8; i++)
    {
        std::cout << std::setw(2) << i << ' ';
        std::cout << std::setw(6) << a.m_I64[i] << ' ';
        std::cout << std::setw(6) << b.m_I64[i] << ' ';
        std::cout << std::setw(8) << c[0].m_I64[i] << ' ';
        std::cout << std::setw(8) << c[1].m_I64[i] << ' ';
        std::cout << std::setw(8) << c[2].m_I64[i] << ' ';
        std::cout << std::setw(8) << c[3].m_I64[i] << ' ';
        std::cout << std::setw(8) << c[4].m_I64[i] << ' ';
        std::cout << std::setw(8) << c[5].m_I64[i] << '\n';
    }
}

void MathI16_Iavx512(ZmmVal c[6], const ZmmVal* a, const ZmmVal* b)
{
    __m512i a_vals = _mm512_load_si512((__m512i*)a);
    __m512i b_vals = _mm512_load_si512((__m512i*)b);

    __m512i c_vals0 = _mm512_add_epi16(a_vals, b_vals);
    __m512i c_vals1 = _mm512_adds_epi16(a_vals, b_vals);
    __m512i c_vals2 = _mm512_sub_epi16(a_vals, b_vals);
    __m512i c_vals3 = _mm512_subs_epi16(a_vals, b_vals);
    __m512i c_vals4 = _mm512_min_epi16(a_vals, b_vals);
    __m512i c_vals5 = _mm512_max_epi16(a_vals, b_vals);

    _mm512_store_si512((__m512i*)&c[0], c_vals0);
    _mm512_store_si512((__m512i*)&c[1], c_vals1);
    _mm512_store_si512((__m512i*)&c[2], c_vals2);
    _mm512_store_si512((__m512i*)&c[3], c_vals3);
    _mm512_store_si512((__m512i*)&c[4], c_vals4);
    _mm512_store_si512((__m512i*)&c[5], c_vals5);
}

void MathI64_Iavx512(ZmmVal c[6], const ZmmVal* a, const ZmmVal* b)
{
    __m512i a_vals = _mm512_load_si512((__m512i*)a);
    __m512i b_vals = _mm512_load_si512((__m512i*)b);

    __m512i c_vals0 = _mm512_add_epi64(a_vals, b_vals);
    __m512i c_vals1 = _mm512_sub_epi64(a_vals, b_vals);
    __m512i c_vals2 = _mm512_mullo_epi64(a_vals, b_vals);
    __m512i c_vals3 = _mm512_sllv_epi64(a_vals, b_vals);
    __m512i c_vals4 = _mm512_srav_epi64(a_vals, b_vals);
    __m512i c_vals5 = _mm512_abs_epi64(a_vals);

    _mm512_store_si512((__m512i*)&c[0], c_vals0);
    _mm512_store_si512((__m512i*)&c[1], c_vals1);
    _mm512_store_si512((__m512i*)&c[2], c_vals2);
    _mm512_store_si512((__m512i*)&c[3], c_vals3);
    _mm512_store_si512((__m512i*)&c[4], c_vals4);
    _mm512_store_si512((__m512i*)&c[5], c_vals5);
}
