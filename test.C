#include<iostream>
#include<cstdlib>

#include "openmp_rng.h"

int main()
{
//  {	
//    size_t sz = 1024 * 1024 * 128;
//    unsigned int* data = (unsigned int*)malloc(sizeof(unsigned int) * sz);
//  
//    unsigned int* data_d;
//    hipMalloc((void**)&data_d, sizeof(unsigned int) * sz);
//  
//    run(data_d, ROCRAND_RNG_PSEUDO_XORWOW, sz);
//  
//    hipMemcpy(data, data_d, sizeof(unsigned int) * sz, hipMemcpyDeviceToHost);
//    for(int i=0; i<sz; i+=sz/20)
//      std::cout << data[i] << std::endl;
//  }
  {	
    size_t sz = 1024 * 1024 * 128;
    unsigned int* data = (unsigned int*)malloc(sizeof(unsigned int) * sz);

#pragma omp target data map(tofrom:data[0:sz])
    {
#pragma omp target data use_device_ptr(data)	    
      {
        openmp_get_rng_uniform_uint(data, sz);
      }
    }
  
    for(int i=0; i<sz; i+=sz/20)
      std::cout << data[i] << std::endl;
  }

}
