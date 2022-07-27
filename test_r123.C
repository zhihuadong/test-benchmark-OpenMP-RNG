#include<iostream>
#include<cstdlib>
#include<vector>

#include "openmp_rng.h"

#define MYMALLOC(T, S) T* data = (T*)malloc(sizeof(T) * (S));

int main()
{
  generator_enum gen_type = generator_enum::philox;
//  {	
//    size_t sz = 216;
//    MYMALLOC(unsigned int, sz);
//    {
//      omp_get_rng_uniform_uint(data, sz, 1234ull, gen_type);
//    }
//  
//    for(int i=0; i<sz; i++)
//      std::cout << "i = " << i << "\t data = " << data[i] << std::endl;
//  }
//  {	
//    size_t sz = 216;
//    MYMALLOC(float, sz);
//    {
//      omp_get_rng_uniform_float(data, sz, 1234ull, gen_type);
//    }
//  
//    for(int i=0; i<sz; i++)
//      std::cout << "i = " << i << "\t data = " << data[i] << std::endl;
//  }
//  {	
//    size_t sz = 216;
//    MYMALLOC(double, sz);
//    {
//      omp_get_rng_uniform_double(data, sz, 1234ull, gen_type);
//    }
//
//    for(int i=0; i<sz; i++)
//      std::cout << "i = " << i << "\t data = " << data[i] << std::endl;
//  }
//  {	
//    size_t sz = 216;
//    MYMALLOC(float, sz);
//    {
//      omp_get_rng_normal_float(data, sz, 0.0f, 1.0f, 1234ull, gen_type);
//    }
//    for(int i=0; i<sz; i++)
//      std::cout << "i = " << i << "\t data = " << data[i] << std::endl;
//  }
  {	
    size_t sz = 216;
    MYMALLOC(double, sz);
    {
      omp_get_rng_normal_double(data, sz, 0.0, 10.0, 1234ull, gen_type);
    }
    for(int i=0; i<sz; i++)
      std::cout << "i = " << i << "\t data = " << data[i] << std::endl;
  }

}
