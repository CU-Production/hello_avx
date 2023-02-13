//
// Created by Admin on 2023/2/13.
//

// AVX header
#include <immintrin.h>
#include <iostream>

void CalcZ_Cpp(float* z, const float* x, const float* y, size_t len)
{
    for (size_t i = 0; i < len; ++i)
    {
        z[i] = x[i] + y[i];
    }
}

void CalcZ_Iavx(float* z, const float* x, const float* y, size_t len)
{
    size_t i = 0;
    const size_t num_simd_elements = 8;

    for(; i + num_simd_elements < len; i += num_simd_elements)
    {
        // Calculate z[i:i+7] = x[i:i+7] + y[i:i+7]
        __m256 x_vals = _mm256_loadu_ps(&x[i]);
        __m256 y_vals = _mm256_loadu_ps(&y[i]);
        __m256 z_vals = _mm256_add_ps(x_vals, y_vals);

        _mm256_storeu_ps(&z[i], z_vals);
    }

    // Calcuate z[i] = x[i] + y[i] for any remaining elements
    for(; i < len; ++i)
    {
        z[i] = x[i] + y[i];
    }
}

int main()
{
    float x[] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20};
    float y[] = {1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000};

    float z[20] = {};

#pragma region Cpp

    CalcZ_Cpp(z, x, y, 20);

    // Print z[]
    std::cout << "Begin print z[]" << std::endl;
    for (int i = 0; i < 20; ++i)
    {
        std::cout << "\tz[" << i << "]: " << z[i] << std::endl;
    }
    std::cout << "End print z[]" << std::endl;

#pragma endregion Cpp

    // Clear z
    for (float & i : z)
    {
        i = 0;
    }

#pragma region AVX

    CalcZ_Iavx(z, x, y, 20);

    // Print z[]
    std::cout << "Begin print z[]" << std::endl;
    for (int i = 0; i < 20; ++i)
    {
        std::cout << "\tz[" << i << "]: " << z[i] << std::endl;
    }
    std::cout << "End print z[]" << std::endl;

#pragma endregion AVX

    return 0;
}
