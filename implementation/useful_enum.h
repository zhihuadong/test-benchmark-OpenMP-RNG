#ifndef _USEFUL_ENUM_
#define _USEFUL_ENUM_

enum class generator_enum
{
  philox,
  xorwow,
  mrg32k3a,
  sobol32,
  sobol64,
  mtgp32
};

/* For each backend, below is the support table of each generator
 *
 * generator	random123	CUDA		HIP
 * philox	True		True		True
 * xorwow	False		True		True
 * mrg32k3a	False		True		True
 * sobol32	False		True		True
 * sobol64	False		True		True
 * mtgp32	False		True		True
 *
 */

enum class distribution_enum
{
  uniform_uint,
  uniform_ull,
  uniform_float,
  uniform_double,
  normal_float,
  normal_double
};

#endif
