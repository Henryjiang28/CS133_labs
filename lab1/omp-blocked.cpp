#include <cmath>
#include <cstring>

#include <chrono>
#include <iostream>
#include <random>
#include <vector>
#include <omp.h>

#include "lib/gemm.h"
#define BLOCK_I 64
#define BLOCK_J 64
#define BLOCK_K 64

// Using declarations, if any...

void GemmParallelBlocked(const float a[kI][kK], const float b[kK][kJ],
                         float c[kI][kJ]) {


  for (int i = 0; i < kI; ++i) {
      std::memset(c[i], 0, sizeof(float) * kJ);
    }

    #pragma omp parallel for collapse(2)
    for (int ii = 0; ii < kI; ii += BLOCK_I) {
      for (int jj = 0; jj < kJ; jj += BLOCK_J) {
        for (int kk= 0; kk < kK; kk += BLOCK_K) {
            for (int i = ii; i < std::min(ii + BLOCK_I, kI); ++i) {
              for (int k = kk; k < std::min(kk + BLOCK_J, kK); ++k) {
                for (int j = jj; j < std::min(jj + BLOCK_K, kJ); ++j) {
                  c[i][j] += a[i][k] * b[k][j];
                }
              }
            }
          }
        }
      }
    }



