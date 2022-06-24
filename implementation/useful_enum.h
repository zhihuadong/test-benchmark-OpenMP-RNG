#ifndef _USEFUL_ENUM_
#define _USEFUL_ENUM_

#include<cassert>

/* For each backend, below is the support table of each generator
 *
 * generator	basic		random123	CUDA		HIP
 * philox	False		True		True		True
 * xorwow	False		False		True		True
 * mrg32k3a	False		False		True		True
 * sobol32	False		False		True		True
 * sobol64	False		False		True		True
 * mtgp32	False		False		True		True
 * mt19937    	True		False       	True    	False
 *
 */

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

inline std::ostream& operator<<(std::ostream & os, const generator_enum & gen)
{
  switch(gen)
  {
    case generator_enum::philox:
      os << "philox";
      break;
    case generator_enum::xorwow:
      os << "xorwow";
      break;
    case generator_enum::mrg32k3a:
      os << "mrg32k3a";
      break;
    case generator_enum::sobol32:
      os << "sobol32";
      break;
    case generator_enum::sobol64:
      os << "sobol64";
      break;
    case generator_enum::mtgp32:
      os << "mtgp32";
      break;
    case generator_enum::mt19937:
      os << "mt19937";
      break;
    default:
      os << "Error: Unrecognized generator_enum";
      assert(0);
  }
  return os;
}

#endif
