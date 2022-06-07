# test-benchmark-OpenMP-RNG
Test and benchmark different random number generator in OpenMP.

This is the first version, and it is designed as a header-only library for Random Number Generators for OpenMP.

It should include the following component:\
1). Main body of library\
2). Compiling command for different compiler

It should support the following architechture:\
1). CPU serial\
2). NVIDIA GPU\
3). AMD GPU\
4). (Later) Intel GPU

It should be based on the following library:\
1). Random123. We should use that at least on CPU\
2). curand. We want to use that on NVIDIA GPU, and need to test if its performance is better than Random123\
3). rocrand. We want to use that on AMD GPU, and need to test if its performance is better than Random123\
4). (Later) Intel vendor RNG library

It should implement at least the following distributions of random numbers:\
1). Uniform distribution between [a,b]: U(a,b) (inclusive/exclusive?)\
2). Gaussian distribution with center and sigma: f(mu, sigma)\
3). (Later) Binomial distribution with probability p and number of trials n: P(n,p)


Some important notes:
1). cuRAND host APIs are thread safe as long as different host threads use different generators, generators are not MT19937 (CURAND\_RNG\_PSEUDO\_MT19937), and the outputs are disjoint.
2). To get the best performance for cuRAND host APIs users are encouraged to use CURAND\_ORDERING\_PSEUDO\_BEST or CURAND\_ORDERING\_PSEUDO\_DYNAMIC orderings. #TODO
3). Different seeds will produce different sequences. Different seeds will often not be statistically correlated with each other, but some pairs of seed values may generate sequences which are statistically correlated.
