#ifndef _OPENMP_RNG_RANDOM123_H
#define _OPENMP_RNG_RANDOM123_H

#include <iostream>
#include <cassert>

#include <omp.h>

#include "Random123/philox.h"
#include "Random123/features/compilerfeatures.h"
#include "Random123/uniform.hpp"
#include "Random123/boxmuller.hpp"

#include "useful_enum.h"

void _getThreadWork(int &start, int &end, int tot_work, int tid, int tot_thread)
{
  start = ((tot_work + tot_thread - 1) / tot_thread) * tid;
  end = start + (tot_work + tot_thread - 1)/ tot_thread;
  if(end > tot_work)
    end = tot_work;
}

void omp_get_rng_uniform_uint(unsigned int* data_h,
                              const size_t sz,
                              unsigned long long seed,
                              const generator_enum rng_type_enum,
                              const size_t offset, const size_t dimensions)
{
  assert(rng_type_enum == generator_enum::philox && "Error: OpenMP-RNG-RANDOM123 only support philox generator type!\n");
  assert(!(sz % 4) && "Error: OpenMP-RNG-RANDOM123 only support size of random numbers to be a multiple of 4!\n");

  typedef r123::Philox4x32 CBRNG;
  int nt;
#pragma omp parallel private(nt)
  {
    nt = omp_get_num_threads();
    int id = omp_get_thread_num();

    CBRNG g;
    CBRNG::key_type key = {{(uint32_t)id, (uint32_t)seed}};
    CBRNG::ctr_type ctr = {{(uint32_t)offset}};

    int start = 0, end = 0;
    _getThreadWork(start, end, sz/4, id, nt);

    for(int i=start; i<end; ++i)
    {
      CBRNG::ctr_type rand = g(ctr, key);
      data_h[4*i]   = rand[0];
      data_h[4*i+1] = rand[1];
      data_h[4*i+2] = rand[2];
      data_h[4*i+3] = rand[3];
      ctr.incr();
    }
  }
}

void omp_get_rng_uniform_float(float* data_h,
                               const size_t sz,
                               unsigned long long seed,
                               const generator_enum rng_type_enum,
                               const size_t offset, const size_t dimensions)
{
  assert(rng_type_enum == generator_enum::philox && "Error: OpenMP-RNG-RANDOM123 only support philox generator type!\n");
  assert(!(sz % 4) && "Error: OpenMP-RNG-RANDOM123 only support size of random numbers to be a multiple of 4!\n");

  typedef r123::Philox4x32 CBRNG;
  int nt;
#pragma omp parallel private(nt)
  {
    nt = omp_get_num_threads();
    int id = omp_get_thread_num();

    CBRNG g;
    CBRNG::key_type key = {{(uint32_t)id, (uint32_t)seed}};
    CBRNG::ctr_type ctr = {{(uint32_t)offset}};

    int start = 0, end = 0;
    _getThreadWork(start, end, sz/4, id, nt);

    for(int i=start; i<end; ++i)
    {
      CBRNG::ctr_type rand = g(ctr, key);
      data_h[4*i]   = r123::u01<float>(rand[0]);
      data_h[4*i+1] = r123::u01<float>(rand[1]);
      data_h[4*i+2] = r123::u01<float>(rand[2]);
      data_h[4*i+3] = r123::u01<float>(rand[3]);
      ctr.incr();
    }
  }
}

void omp_get_rng_uniform_double(double* data_h,
                                const size_t sz,
                                unsigned long long seed,
                                const generator_enum rng_type_enum,
                                const size_t offset, const size_t dimensions)
{
  assert(rng_type_enum == generator_enum::philox && "Error: OpenMP-RNG-RANDOM123 only support philox generator type!\n");
  assert(!(sz % 4) && "Error: OpenMP-RNG-RANDOM123 only support size of random numbers to be a multiple of 4!\n");

  typedef r123::Philox4x64 CBRNG;
  int nt;
#pragma omp parallel private(nt)
  {
    nt = omp_get_num_threads();
    int id = omp_get_thread_num();

    CBRNG g;
    CBRNG::key_type key = {{(uint64_t)id, (uint64_t)seed}};
    CBRNG::ctr_type ctr = {{(uint64_t)offset}};

    int start = 0, end = 0;
    _getThreadWork(start, end, sz/4, id, nt);

    for(int i=start; i<end; ++i)
    {
      CBRNG::ctr_type rand = g(ctr, key);
      data_h[4*i]   = r123::u01<double>(rand[0]);
      data_h[4*i+1] = r123::u01<double>(rand[1]);
      data_h[4*i+2] = r123::u01<double>(rand[2]);
      data_h[4*i+3] = r123::u01<double>(rand[3]);
      ctr.incr();
    }
  }
}

void _boxmuller(uint32_t u0, uint32_t u1, float &r1, float &r2)
{
  float uf0 = r123::u01<float>(u0);
  float uf1 = r123::u01<float>(u1);

  const float PIf = 3.1415926535897932f;

  r1 = sinf(2.0f * PIf * uf0);
  r2 = cosf(2.0f * PIf * uf0);

  float r = sqrtf(-2.0f * logf(uf1));
  r1 *= r;
  r2 *= r;
}

void _boxmuller(uint64_t u0, uint64_t u1, double &r1, double &r2)
{
  double ud0 = r123::u01<double>(u0);
  double ud1 = r123::u01<double>(u1);

  r1 = sin(2.0 * M_PI * ud0);
  r2 = cos(2.0 * M_PI * ud0);

  double r = sqrt(-2.0 * log(ud1));
  r1 *= r;
  r2 *= r;
}

void omp_get_rng_normal_float(float* data_h,
                              const size_t sz,
                              float mean, float stddev,
                              unsigned long long seed,
                              const generator_enum rng_type_enum,
                              const size_t offset, const size_t dimensions)
{
  assert(rng_type_enum == generator_enum::philox && "Error: OpenMP-RNG-RANDOM123 only support philox generator type!\n");
  assert(!(sz % 4) && "Error: OpenMP-RNG-RANDOM123 only support size of random numbers to be a multiple of 4!\n");

  typedef r123::Philox4x32 CBRNG;
  int nt;
#pragma omp parallel private(nt)
  {
    nt = omp_get_num_threads();
    int id = omp_get_thread_num();

    CBRNG g;
    CBRNG::key_type key = {{(uint32_t)id, (uint32_t)seed}};
    CBRNG::ctr_type ctr = {{(uint32_t)offset}};

    int start = 0, end = 0;
    _getThreadWork(start, end, sz/4, id, nt);

    for(int i=start; i<end; ++i)
    {
      CBRNG::ctr_type rand = g(ctr, key);
      _boxmuller(rand[0], rand[1], data_h[4*i],   data_h[4*i+1]);
      _boxmuller(rand[2], rand[3], data_h[4*i+2], data_h[4*i+3]);
      if(mean != 0.0f || stddev != 1.0f)
      {
        data_h[4*i]   = ( data_h[4*i]   * stddev ) + mean;
        data_h[4*i+1] = ( data_h[4*i+1] * stddev ) + mean;
        data_h[4*i+2] = ( data_h[4*i+2] * stddev ) + mean;
        data_h[4*i+3] = ( data_h[4*i+3] * stddev ) + mean;
      }
      ctr.incr();
    }
  }
}

void omp_get_rng_normal_double(double* data_h,
                               const size_t sz,
                               double mean, double stddev,
                               unsigned long long seed,
                               const generator_enum rng_type_enum,
                               const size_t offset, const size_t dimensions)
{
  assert(rng_type_enum == generator_enum::philox && "Error: OpenMP-RNG-RANDOM123 only support philox generator type!\n");
  assert(!(sz % 4) && "Error: OpenMP-RNG-RANDOM123 only support size of random numbers to be a multiple of 4!\n");

  typedef r123::Philox4x64 CBRNG;
  int nt;
#pragma omp parallel private(nt)
  {
    nt = omp_get_num_threads();
    int id = omp_get_thread_num();

    CBRNG g;
    CBRNG::key_type key = {{(uint64_t)id, (uint64_t)seed}};
    CBRNG::ctr_type ctr = {{(uint64_t)offset}};

    int start = 0, end = 0;
    _getThreadWork(start, end, sz/4, id, nt);

    for(int i=start; i<end; ++i)
    {
      CBRNG::ctr_type rand = g(ctr, key);
      _boxmuller(rand[0], rand[1], data_h[4*i],   data_h[4*i+1]);
      _boxmuller(rand[2], rand[3], data_h[4*i+2], data_h[4*i+3]);
      if(mean != 0.0 || stddev != 1.0)
      {
        data_h[4*i]   = ( data_h[4*i]   * stddev ) + mean;
        data_h[4*i+1] = ( data_h[4*i+1] * stddev ) + mean;
        data_h[4*i+2] = ( data_h[4*i+2] * stddev ) + mean;
        data_h[4*i+3] = ( data_h[4*i+3] * stddev ) + mean;
      }
      ctr.incr();
    }
  }
}

#endif
