clang++ -O3 -fopenmp -fopenmp-targets=nvptx64-nvidia -Xopenmp-target -march=sm_70 benchmark.C -lcurand -DARCH_CUDA
#echo 'Before running the program, do export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm-4.5.2/lib/'
