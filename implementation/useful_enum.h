#ifndef _USEFUL_ENUM_
#define _USEFUL_ENUM_

enum class generator_enum
{
  philox,
  xorwow,
  mrg32k3a,
  sobol32,
  sobol64,
  mtgp32,
  mt19937
};

/* For each backend, below is the support table of each generator
 *
 * generator	random123	  CUDA		HIP
 * philox	    True		    True		True
 * xorwow	    False		    True		True
 * mrg32k3a	  False		    True		True
 * sobol32	  False		    True		True
 * sobol64	  False		    True		True
 * mtgp32	    False		    True		True
 * mt19937    False       True    False
 *
 */

#endif
