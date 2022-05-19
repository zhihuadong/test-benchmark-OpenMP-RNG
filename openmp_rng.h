#ifndef _OPENMP_RNG_IMPL
#define _OPENMP_RNG_IMPL

//Options for _RNG_impl:
//RNG_IMPL_BASIC             :not using any external libraries, only use std::random for CPU
//RNG_IMPL_RANDOM123:        :using RANDOM123 library for CPU
//RNG_IMPL_CURAND            :using CUDA CURAND library for NVIDIA GPU
//RNG_IMPL_ROCRAND           :using HIP ROCRAND library for AMD GPU


//FIXME: Later need to find the best flag/macro to be used

#if defined(USE_RANDOM123)
//Use ramdom123 version, make sure include random123 (no need to link since it is header-only)
#define RNG_IMPL_RANDOM123

#elif defined(ARCH_CUDA)
//Use cuda version, make sure to link curand
#define RNG_IMPL_CURAND

#elif defined(ARCH_HIP)
//Use hip version, make sure to link rocrand
#define RNG_IMPL_ROCRAND

#else
//Default to std::random version (not parallized!)
#define RNG_IMPL_BASIC
#warning "Since we did not specify the RNG to use, we use std::random, which is not and should not be parallized!"

#endif  


#if defined(RNG_IMPL_RANDOM123)
#  include "implementation/"
#elif defined(RNG_IMPL_CURAND)
#  include "implementation/"
#elif defined(RNG_IMPL_ROCRAND)
#  include "implementation/"
#elif defined(RNG_IMPL_BASIC)
#  include "implementation/"
#else
#  error Must specify a RNG_IMPL_* in openmp_rng.h
#endif

//#FIXME what is the set of arguments do we need?
template<typename T>
void omp_generate_rng(T* data,	
		      const distribution_type dis,
		      const generator_type gen,
		      size_t num,
		      size_t seed,
		      size_t offset,
		      size_t dimension)
{
	_openmp_rng_impl::do_it();
}

#endif
