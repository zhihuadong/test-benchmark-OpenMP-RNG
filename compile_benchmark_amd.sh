/opt/rocm-4.5.2/llvm/bin/clang++ -O3 -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx906 benchmark.C -L/opt/rocm-4.5.2/rocrand/lib/ -lrocrand -DARCH_HIP 
echo 'Before running the program, do export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm-4.5.2/lib/'
