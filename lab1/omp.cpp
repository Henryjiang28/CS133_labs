#include <cmath>
#include <cstring>

#include <chrono>
#include <iostream>
#include <random>
#include <vector>
#include <omp.h>

#include "lib/gemm.h"

// Using declarations, if any...

void GemmParallel(const float a[kI][kK], const float b[kK][kJ],
                  float c[kI][kJ]) {
    for (int i = 0; i < kI; ++i) {
      std::memset(c[i], 0, sizeof(float) * kJ);
    }

    int i, j, k;
    #pragma omp parallel for collapse(2)
    for (i = 0; i < kI; ++i) {
      for (k = 0; k < kK; ++k) {
        for (j = 0; j < kJ; ++j) {
          c[i][j] += a[i][k] * b[k][j];
        }
      }
    }
  }

  // void GemmParallel(const float a[kI][kK], const float b[kK][kJ],
  //                 float c[kI][kJ]) {
  //   for (int i = 0; i < kI; ++i) {
  //     std::memset(c[i], 0, sizeof(float) * kJ);
  //   }

  
  //   for (int i = 0; i < kI; ++i) {
  //       for (int j = 0; j < kJ; ++j) {
  //         for (int k = 0; k < kK; ++k) {
  //           c[i][j] += a[i][k] * b[k][j];
  //       }
  //     }
  //   }
  // }


