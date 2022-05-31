#ifndef _OPENMP_RNG_CURAND_H
#define _OPENMP_RNG_CURAND_H

#include <iostream>
#include <cuda_runtime.h>
#include <curand.h>

#include "useful_enum.h"

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    exit(EXIT_FAILURE);}} while(0)
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    exit(EXIT_FAILURE);}} while(0)

inline curandRngType get_rng_type(const generator_enum rng_type_enum)
{
  curandRngType rng_type = ROCRAND_RNG_PSEUDO_XORWOW;
  switch(rng_type_enum)
  {
    case generator_enum::philox:
      rng_type = CURAND_RNG_PSEUDO_PHILOX4_32_10;
      break;
    case generator_enum::xorwow:
      rng_type = CURAND_RNG_PSEUDO_XORWOW;
      break;
    case generator_enum::mrg32k3a:
      rng_type = CURAND_RNG_PSEUDO_MRG32K3A;
      break;
    case generator_enum::sobol32:
      rng_type = CURAND_RNG_QUASI_SOBOL32;
      break;
    case generator_enum::mtgp32:
      rng_type = CURAND_RNG_PSEUDO_MTGP32;
      break;
    default:
      assert(0 && "Error: rng_type set incorrectly, can not find desired rng_type_enum.\n");      
  }
  return rng_type;
}

void openmp_get_rng_uniform_uint(unsigned int* data_d, 
         			 const size_t sz, 
         			 const generator_enum rng_type_enum = generator_enum::philox,
         			 const size_t offset = 0, const size_t dimensions = 1)
{
  curandGenerator_t generator;	

  curandRngType rng_type = get_rng_type(rng_type_enum);

  CURAND_CALL(curandCreateGenerator(&generator, rng_type));

  curandStatus_t status = curandSetQuasiRandomGeneratorDimensions(generator, dimensions);
  if (status != CURAND_STATUS_TYPE_ERROR) // If the RNG is not quasi-random
  {
      CURAND_CALL(status);
  }

  status = curandSetGeneratorOffset(generator, offset);
  if (status != CURAND_STATUS_TYPE_ERROR) // If the RNG is not pseudo-random
  {
      CURAND_CALL(status);
  }

  CURAND_CALL(curandGenerate(generator, data_d, sz));
  CUDA_CALL(cudaDeviceSynchronize());			//FIXME: might not needed!

  CURAND_CALL(curandDestroyGenerator(generator));
}


#endif
