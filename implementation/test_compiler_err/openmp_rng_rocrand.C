#include <iostream>
#include <hip/hip_runtime.h>
#include <rocrand/rocrand.h>

#include "run.h"

#define HIP_CHECK(condition)         \
  {                                  \
    hipError_t error = condition;    \
    if(error != hipSuccess){         \
        std::cout << "HIP error: " << error << " line: " << __LINE__ << std::endl; \
        exit(error); \
    } \
  }

#define ROCRAND_CHECK(condition)                 \
  {                                              \
    rocrand_status _status = condition;           \
    if(_status != ROCRAND_STATUS_SUCCESS) {       \
        std::cout << "ROCRAND error: " << _status << " line: " << __LINE__ << std::endl; \
        exit(_status); \
    } \
  }

void run(unsigned int* data_d, 
         const size_t sz, 
         const size_t offset, const size_t dimensions)
{
  const rocrand_rng_type rng_type = ROCRAND_RNG_PSEUDO_XORWOW;
  rocrand_generator generator;
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
  HIP_CHECK(hipDeviceSynchronize());

  ROCRAND_CHECK(rocrand_destroy_generator(generator));
}


