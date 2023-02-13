//
// Created by Admin on 2023/2/13.
// Integer Multiplication
//

#include "XmmVal.h"
#include <immintrin.h> // AVX header
#include <iostream>

void MulI16_Iavx(XmmVal c[2], const XmmVal* a, const XmmVal* b);
void MulI32a_Iavx(XmmVal* c, const XmmVal* a, const XmmVal* b);
void MulI32b_Iavx(XmmVal c[2], const XmmVal* a, const XmmVal* b);

static void MulI16(void);
static void MulI32a(void);
static void MulI32b(void);

int main()
{
    const char nl = '\n';
    std::string sep(75, '-');

    MulI16();
    std::cout << nl << sep << nl;
    MulI32a();
    std::cout << nl << sep << nl;
    MulI32b();
    return 0;
}

static void MulI16(void)
{
    const char nl = '\n';
    XmmVal a, b, c[2];

    a.m_I16[0] = 10;        b.m_I16[0] = -5;
    a.m_I16[1] = 3000;      b.m_I16[1] = 100;
    a.m_I16[2] = -2000;     b.m_I16[2] = -9000;
    a.m_I16[3] = 42;        b.m_I16[3] = 1000;
    a.m_I16[4] = -5000;     b.m_I16[4] = 25000;
    a.m_I16[5] = 8;         b.m_I16[5] = 16384;
    a.m_I16[6] = 10000;     b.m_I16[6] = 3500;
    a.m_I16[7] = -60;       b.m_I16[7] = 6000;

    MulI16_Iavx(c, &a, &b);

    std::cout << "\nResults for MulI16_Iavx\n";
    for (size_t i = 0; i < 8; i++)
    {
        std::cout << "a[" << i << "]: " << std::setw(8) << a.m_I16[i] << "  ";
        std::cout << "b[" << i << "]: " << std::setw(8) << b.m_I16[i] << "  ";

        if (i < 4)
        {
            std::cout << "c[0][" << i << "]: ";
            std::cout << std::setw(12) << c[0].m_I32[i] << nl;
        }
        else
        {
            std::cout << "c[1][" << i - 4 << "]: ";
            std::cout << std::setw(12) << c[1].m_I32[i - 4] << nl;
        }
    }
}

static void MulI32a(void)
{
    const char nl = '\n';
    XmmVal a, b, c;

    a.m_I32[0] = 10;        b.m_I32[0] = -500;
    a.m_I32[1] = 3000;      b.m_I32[1] = 100;
    a.m_I32[2] = -2000;     b.m_I32[2] = -12000;
    a.m_I32[3] = 4200;      b.m_I32[3] = 1000;

    MulI32a_Iavx(&c, &a, &b);

    std::cout << "\nResults for MulI32a_Iavx\n";
    for (size_t i = 0; i < 4; i++)
    {
        std::cout << "a[" << i << "]: " << std::setw(10) << a.m_I32[i] << "  ";
        std::cout << "b[" << i << "]: " << std::setw(10) << b.m_I32[i] << "  ";
        std::cout << "c[" << i << "]: " << std::setw(10) << c.m_I32[i] << nl;
    }
}

static void MulI32b(void)
{
    const char nl = '\n';
    XmmVal a, b, c[2];

    a.m_I32[0] = 10;        b.m_I32[0] = -500;
    a.m_I32[1] = 3000;      b.m_I32[1] = 100;
    a.m_I32[2] = -40000;    b.m_I32[2] = -120000;
    a.m_I32[3] = 4200;      b.m_I32[3] = 1000;

    MulI32b_Iavx(c, &a, &b);

    std::cout << "\nResults for MulI32b_Iavx\n";
    for (size_t i = 0; i < 4; i++)
    {
        std::cout << "a[" << i << "]: " << std::setw(10) << a.m_I32[i] << "  ";
        std::cout << "b[" << i << "]: " << std::setw(10) << b.m_I32[i] << "  ";

        if (i < 2)
        {
            std::cout << "c[0][" << i << "]: ";
            std::cout << std::setw(14) << c[0].m_I64[i] << nl;
        }
        else
        {
            std::cout << "c[1][" << i - 2 << "]: ";
            std::cout << std::setw(14) << c[1].m_I64[i - 2] << nl;
        }
    }
}

void MulI16_Iavx(XmmVal c[2], const XmmVal* a, const XmmVal* b)
{
    __m128i a_vals = _mm_load_si128((__m128i*)a);
    __m128i b_vals = _mm_load_si128((__m128i*)b);

    __m128i temp_lo = _mm_mullo_epi16(a_vals, b_vals);
    __m128i temp_hi = _mm_mulhi_epi16(a_vals, b_vals);

    __m128i result_lo = _mm_unpacklo_epi16(temp_lo, temp_hi);
    __m128i result_hi = _mm_unpackhi_epi16(temp_lo, temp_hi);

    _mm_store_si128((__m128i*)&c[0], result_lo);
    _mm_store_si128((__m128i*)&c[1], result_hi);
}

void MulI32a_Iavx(XmmVal* c, const XmmVal* a, const XmmVal* b)
{
    __m128i a_vals = _mm_load_si128((__m128i*)a);
    __m128i b_vals = _mm_load_si128((__m128i*)b);
    __m128i c_vals = _mm_mullo_epi32(a_vals, b_vals);

    _mm_store_si128((__m128i*)c, c_vals);
}

void MulI32b_Iavx(XmmVal c[2], const XmmVal* a, const XmmVal* b)
{
    __m128i a_vals = _mm_load_si128((__m128i*)a);
    __m128i b_vals = _mm_load_si128((__m128i*)b);

    __m128i temp1 = _mm_mul_epi32(a_vals, b_vals);     // q2 | q0
    __m128i temp2 = _mm_srli_si128(a_vals, 4);
    __m128i temp3 = _mm_srli_si128(b_vals, 4);
    __m128i temp4 = _mm_mul_epi32(temp2, temp3);       // q3 | q1

    *(&c[0].m_I64[0]) = _mm_extract_epi64(temp1, 0);    // q0
    *(&c[0].m_I64[1]) = _mm_extract_epi64(temp4, 0);    // q1
    *(&c[1].m_I64[0]) = _mm_extract_epi64(temp1, 1);    // q2
    *(&c[1].m_I64[1]) = _mm_extract_epi64(temp4, 1);    // q3
}
