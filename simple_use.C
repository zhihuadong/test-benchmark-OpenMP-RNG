#include<iostream>
#include<cstdlib>
#include<vector>

#include "openmp_rng.h"

#define MYMALLOC(T, S) T* data = (T*)malloc(sizeof(T) * (S));

int main()
{
#if defined(ARCH_CUDA) || defined(ARCH_HIP) || defined(USE_RANDOM123)
  generator_enum gen_type = generator_enum::philox;
//  generator_enum gen_type = generator_enum::xorwow;
#else
  generator_enum gen_type = generator_enum::mt19937;
#endif
  std::cout << "Generator used is " << gen_type << std::endl;
  std::cout << "Test normal_float" << std::endl;

  size_t sz = 1024 * 1024 * 128;
  size_t output_interval = 32;
  double tt;
  MYMALLOC(float, sz);
  bool test_omp_target_map = true;
  bool test_omp_target_alloc = false;

  //Test omp target map
  if(test_omp_target_map)
  {
#if defined(ARCH_CUDA) || defined(ARCH_HIP)
  #pragma omp target data map(tofrom:data[0:sz])
#endif
    {
      tt = -omp_get_wtime();
#if defined(ARCH_CUDA) || defined(ARCH_HIP)
  #pragma omp target data use_device_ptr(data)
#endif
      omp_get_rng_normal_float(data, sz, 0.0f, 1.0f, 1234ull, gen_type);
      tt += omp_get_wtime();
    }
  }
  
  //Test omp_target_alloc
  if(test_omp_target_alloc)
  {	
    int device_id = omp_get_default_device();
    int host_id = omp_get_initial_device();
    float* data_d = (float*)omp_target_alloc(sizeof(float) * sz, device_id);
    {
      tt = -omp_get_wtime();
      omp_get_rng_normal_float(data_d, sz, 0.0f, 1.0f, 1234ull, gen_type);
      tt += omp_get_wtime();
    }
    omp_target_memcpy(data, data_d, sizeof(float) * sz, 0, 0, host_id, device_id);
  }

  std::cout << "Total time for RNG = " << tt * 1e3 << " ms" << std::endl;
  for(int i=0; i<sz; i+=sz/output_interval)
    std::cout << " data " << i << "\t = " << data[i] << std::endl;
}
