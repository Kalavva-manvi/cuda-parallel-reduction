\# Parallel Reduction in CUDA



CUDA Lab Evaluation, Theme 1.



\- GPU: NVIDIA GeForce RTX 5060 Ti

\- Kernels: v1 interleaved, v2 sequential, v3 grid-stride + warp shuffle + atomicAdd, with Thrust as a library reference

\- Timing: CUDA events, average of 10 runs after a warm-up (CPU: average of 5 runs)

\- Every GPU result is checked against the CPU sum before speeds are reported



\## Build and run



nvcc -O2 -std=c++17 -arch=native -Xcompiler "/Zc:preprocessor" -o reduction.exe reduction.cu

reduction.exe 16777216



\## Folders



\- results/ : saved program output for 1M, 16M and 64M elements

\- charts/ : graphs made from the measured results

\- presentation/ : the PowerPoint

