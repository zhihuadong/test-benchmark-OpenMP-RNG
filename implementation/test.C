#include<iostream>
#include<cstdlib>

#include<hip/hip_runtime.h>
#include "openmp_rng_rocrand.h"

int main()
{
  size_t sz = 1024 * 1024 * 128;
  unsigned int* data = (unsigned int*)malloc(sizeof(unsigned int) * sz);

  unsigned int* data_d;
  hipMalloc((void**)&data_d, sizeof(unsigned int) * sz);

  run(data_d, ROCRAND_RNG_PSEUDO_XORWOW, sz);

  hipMemcpy(data, data_d, sizeof(unsigned int) * sz, hipMemcpyDeviceToHost);
  for(int i=0; i<sz; i+=sz/20)
    std::cout << data[i] << std::endl;
}
