#include<iostream>
#include<cstdlib>
#include<vector>
#include<omp.h>

#include "openmp_rng.h"

#define MYMALLOC(T, S) T* data = (T*)malloc(sizeof(T) * (S));

int main()
{

#if defined(ARCH_CUDA)
  std::vector<generator_enum> gen_type = {generator_enum::philox, generator_enum::xorwow, generator_enum::mrg32k3a, generator_enum::sobol32, generator_enum::sobol64, generator_enum::mtgp32, generator_enum::mt19937};
#elif defined(ARCH_HIP)
  std::vector<generator_enum> gen_type = {generator_enum::philox};
#elif defined(USE_RANDOM123)
  std::vector<generator_enum> gen_type = {generator_enum::philox};
#else
  std::vector<generator_enum> gen_type = {generator_enum::mt19937};
#endif

  //Test the most general usage
  for(int i=0; i<20; i++)
  {
    {	
      size_t sz = 512 * 50000;
      MYMALLOC(double, sz);
#if defined(ARCH_CUDA) || defined(ARCH_HIP)
  #pragma omp target data map(tofrom:data[0:sz])
#endif
      {
        double t_temp = -omp_get_wtime();
#if defined(ARCH_CUDA) || defined(ARCH_HIP)
  #pragma omp target data use_device_ptr(data)
#endif
        omp_get_rng_normal_double(data, sz, 0.0, 1.0, 2020);
        t_temp += omp_get_wtime();
        std::cout << "Iter " << i << "\t requires " << t_temp * 1000 << " ms" << std::endl;
      }
    }
  }
}
