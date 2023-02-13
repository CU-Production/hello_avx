//
// Created by Admin on 2023/2/13.
// Integer Subtraction
//

#include "XmmVal.h"
#include <immintrin.h> // AVX header
#include <iostream>

void SubI32_Iavx(XmmVal* c, const XmmVal* a, const XmmVal* b);
void SubI64_Iavx(XmmVal* c, const XmmVal* a, const XmmVal* b);

static void SubI32(void);
static void SubI64(void);

int main()
{
    SubI32();
    SubI64();
    return 0;
}

static void SubI32(void)
{
    XmmVal a, b, c;
    const char nl = '\n';

    a.m_I32[0] = 1000000;     b.m_I32[0] = 100;
    a.m_I32[1] = 200;         b.m_I32[1] = -200;
    a.m_I32[2] = -30;         b.m_I32[2] = 30000;
    a.m_I32[3] = 40000000;    b.m_I32[3] = 5000;

    SubI32_Iavx(&c, &a, &b);

    std::cout << "\nResults for SubI32_Iavx\n";
    std::cout << "a:  " << a.ToStringI32() << nl;
    std::cout << "b:  " << b.ToStringI32() << nl;
    std::cout << "c:  " << c.ToStringI32() << nl;
}

static void SubI64(void)
{
    XmmVal a, b, c;
    const char nl = '\n';

    a.m_I64[0] = 100000000000; b.m_I64[0] = 99;
    a.m_I64[1] = 200;          b.m_I64[1] = 300000000000;

    SubI64_Iavx(&c, &a, &b);

    std::cout << "\nResults for SubI64_Iavx\n";
    std::cout << "a:  " << a.ToStringI64() << nl;
    std::cout << "b:  " << b.ToStringI64() << nl;
    std::cout << "c:  " << c.ToStringI64() << nl;
}

void SubI32_Iavx(XmmVal* c, const XmmVal* a, const XmmVal* b)
{
    __m128i a_vals = _mm_load_si128((__m128i*)a);
    __m128i b_vals = _mm_load_si128((__m128i*)b);
    __m128i c_vals = _mm_sub_epi32(a_vals, b_vals);

    _mm_store_si128((__m128i*)c, c_vals);
}

void SubI64_Iavx(XmmVal* c, const XmmVal* a, const XmmVal* b)
{
    __m128i a_vals = _mm_load_si128((__m128i*)a);
    __m128i b_vals = _mm_load_si128((__m128i*)b);
    __m128i c_vals = _mm_sub_epi64(a_vals, b_vals);

    _mm_store_si128((__m128i*)c, c_vals);
}
