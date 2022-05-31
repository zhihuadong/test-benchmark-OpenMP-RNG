#ifndef _OPENMP_RNG_ROCRAND_H
#define _OPENMP_RNG_ROCRAND_H

#define __HIP_PLATFORM_AMD__

#include <iostream>
#include <rocrand/rocrand.h>

#include "useful_enum.h"

#define ROCRAND_CHECK(condition)                 \
  {                                              \
    rocrand_status _status = condition;           \
    if(_status != ROCRAND_STATUS_SUCCESS) {       \
        std::cout << "ROCRAND error: " << _status << " line: " << __LINE__ << std::endl; \
        exit(_status); \
    } \
  }

inline rocrand_rng_type get_rng_type(const generator_enum rng_type_enum)
{
  rocrand_rng_type rng_type = ROCRAND_RNG_PSEUDO_XORWOW;
  switch(rng_type_enum)
  {
    case generator_enum::philox:
      rng_type = ROCRAND_RNG_PSEUDO_PHILOX4_32_10;
      break;
    case generator_enum::xorwow:
      rng_type = ROCRAND_RNG_PSEUDO_XORWOW;
      break;
    case generator_enum::mrg32k3a:
      rng_type = ROCRAND_RNG_PSEUDO_MRG32K3A;
      break;
    case generator_enum::sobol32:
      rng_type = ROCRAND_RNG_QUASI_SOBOL32;
      break;
    case generator_enum::mtgp32:
      rng_type = ROCRAND_RNG_PSEUDO_MTGP32;
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
  rocrand_generator generator;

  rocrand_rng_type rng_type = get_rng_type(rng_type_enum);

  ROCRAND_CHECK(rocrand_create_generator(&generator, rng_type));

  rocrand_status status = rocrand_set_quasi_random_generator_dimensions(generator, dimensions);
  if (status != ROCRAND_STATUS_TYPE_ERROR) // If the RNG is not quasi-random
  {
      ROCRAND_CHECK(status);
  }

  status = rocrand_set_offset(generator, offset);
  if (status != ROCRAND_STATUS_TYPE_ERROR) // If the RNG is not pseudo-random
  {
      ROCRAND_CHECK(status);
  }

  ROCRAND_CHECK(rocrand_generate(generator, data_d, sz));

  ROCRAND_CHECK(rocrand_destroy_generator(generator));
}


#endif
