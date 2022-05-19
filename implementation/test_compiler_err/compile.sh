hipcc -c openmp_rng_rocrand.C
/opt/rocm/llvm/bin/clang++ -target x86_64-pc-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx906 -c test.C
/opt/rocm/llvm/bin/clang++ -target x86_64-pc-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx906 test.o openmp_rng_rocrand.o -L/opt/rocm-4.5.2/lib/ -lamdhip64 -lrocrand
echo 'Before running ./a.out, do: export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm-4.5.2/lib/'


