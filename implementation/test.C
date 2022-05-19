#include<iostream>
#include<cstdlib>

#include<hip/hip_runtime.h>
#include "openmp_rng_rocrand.h"

int main()
{
  int sz = 1024 * 1024 * 128;
  int* data = (int*)malloc(sizeof(int) * sz);

  int* data_d;
  hipMalloc((void**)&data_d, sizeof(int) * sz);

  run(data_d, ROCRAND_RNG_PSEUDO_XORWOW, sz);

  hipMemcpy(data, data_d, sizeof(int) * sz, hipMemcpyDeviceToHost);
  for(int i=0; i<sz/20; i++)
    std::cout << data[i] << std::endl;
}
