#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>

#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA error: %s\n",cudaGetErrorString(e));exit(1);}}while(0)

// v1: interleaved addressing (divergent)
__global__ void k1(const float* in, float* out, int n){
    extern __shared__ float s[];
    int t = threadIdx.x, i = blockIdx.x*blockDim.x + t;
    s[t] = (i < n) ? in[i] : 0.f;
    __syncthreads();
    for(int st=1; st<blockDim.x; st*=2){
        if(t % (2*st) == 0) s[t] += s[t+st];
        __syncthreads();
    }
    if(t==0) out[blockIdx.x] = s[0];
}

// v2: sequential addressing
__global__ void k2(const float* in, float* out, int n){
    extern __shared__ float s[];
    int t = threadIdx.x, i = blockIdx.x*blockDim.x + t;
    s[t] = (i < n) ? in[i] : 0.f;
    __syncthreads();
    for(int st=blockDim.x/2; st>0; st>>=1){
        if(t < st) s[t] += s[t+st];
        __syncthreads();
    }
    if(t==0) out[blockIdx.x] = s[0];
}

// v3: grid-stride loop + warp shuffle + one atomicAdd per block
__global__ void k3(const float* in, float* out, int n){
    int t = threadIdx.x;
    float v = 0.f;
    for(int i=blockIdx.x*blockDim.x+t; i<n; i+=gridDim.x*blockDim.x) v += in[i];
    for(int o=16; o>0; o>>=1) v += __shfl_down_sync(0xffffffff, v, o);
    __shared__ float w[32];
    int lane = t & 31, wid = t >> 5;
    if(lane==0) w[wid] = v;
    __syncthreads();
    v = (t < blockDim.x/32) ? w[lane] : 0.f;
    if(wid==0){
        for(int o=16; o>0; o>>=1) v += __shfl_down_sync(0xffffffff, v, o);
        if(t==0) atomicAdd(out, v);
    }
}

// multi-pass driver for v1/v2 with a given block size
float run_multipass(void (*kern)(const float*,float*,int), float* d_in, float* d_a, float* d_b, int n, int Bs){
    const float* src = d_in; float* dst = d_a; int m = n;
    while(true){
        int blocks = (m + Bs - 1) / Bs;
        kern<<<blocks, Bs, Bs*sizeof(float)>>>(src, dst, m);
        m = blocks;
        if(m == 1) break;
        src = dst; dst = (dst == d_a) ? d_b : d_a;
    }
    float r; CK(cudaMemcpy(&r, dst, sizeof(float), cudaMemcpyDeviceToHost));
    return r;
}

float run_v3(float* d_in, float* d_out, int n, int Bs){
    CK(cudaMemset(d_out, 0, sizeof(float)));
    k3<<<1024, Bs>>>(d_in, d_out, n);
    float r; CK(cudaMemcpy(&r, d_out, sizeof(float), cudaMemcpyDeviceToHost));
    return r;
}

// warm-up run, then average of 'runs' timed runs using CUDA events
template<typename F>
float timeit(F f, float &result, int runs=10){
    f();
    cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
    float total = 0;
    for(int r=0;r<runs;r++){
        cudaEventRecord(a);
        result = f();
        cudaEventRecord(b); cudaEventSynchronize(b);
        float ms; cudaEventElapsedTime(&ms,a,b); total += ms;
    }
    cudaEventDestroy(a); cudaEventDestroy(b);
    return total/runs;
}

int main(int argc, char** argv){
    int n = (argc>1) ? atoi(argv[1]) : (1<<24);
    cudaDeviceProp p; cudaGetDeviceProperties(&p,0);
    printf("GPU: %s | N = %d\n", p.name, n);
    double bytes = (double)n * sizeof(float);

    std::vector<float> h(n);
    srand(1);
    for(int i=0;i<n;i++) h[i] = (rand()%100)/100.0f;

    // CPU reference: warm-up, then average of 5 timed runs
    double ref = 0;
    for(int i=0;i<n;i++) ref += h[i];
    double cpu_total = 0;
    const int CPU_RUNS = 5;
    for(int r=0;r<CPU_RUNS;r++){
        auto c0 = std::chrono::high_resolution_clock::now();
        double s = 0;
        for(int i=0;i<n;i++) s += h[i];
        auto c1 = std::chrono::high_resolution_clock::now();
        cpu_total += std::chrono::duration<double, std::milli>(c1-c0).count();
        ref = s;
    }
    double cpu_ms = cpu_total / CPU_RUNS;
    printf("CPU ref sum = %.2f\n", ref);
    printf("CPU time (avg of %d runs) = %.4f ms\n", CPU_RUNS, cpu_ms);

    float *d_in,*d_a,*d_b,*d_out;
    int maxPartials = (n + 127) / 128;          // enough for the smallest block size
    CK(cudaMalloc(&d_in,n*sizeof(float)));
    CK(cudaMalloc(&d_a,maxPartials*sizeof(float)));
    CK(cudaMalloc(&d_b,maxPartials*sizeof(float)));
    CK(cudaMalloc(&d_out,sizeof(float)));
    CK(cudaMemcpy(d_in,h.data(),n*sizeof(float),cudaMemcpyHostToDevice));

    auto ok = [&](float r){ return fabs(r-ref)/ref < 1e-3; };

    // ---- Block-size sweep ----
    printf("\n== Block-size sweep (times in ms) ==\n");
    printf("%-8s %10s %10s %10s %12s %8s\n", "Block", "v1", "v2", "v3", "v3 GB/s", "Check");
    int sizes[] = {128, 256, 512, 1024};
    for(int bi=0; bi<4; bi++){
        int Bs = sizes[bi];
        float r1,r2,r3;
        float t1 = timeit([&]{ return run_multipass(k1,d_in,d_a,d_b,n,Bs); }, r1);
        float t2 = timeit([&]{ return run_multipass(k2,d_in,d_a,d_b,n,Bs); }, r2);
        float t3 = timeit([&]{ return run_v3(d_in,d_out,n,Bs); }, r3);
        bool all = ok(r1) && ok(r2) && ok(r3);
        printf("%-8d %10.4f %10.4f %10.4f %12.1f %8s\n", Bs, t1, t2, t3, bytes/(t3*1e6), all ? "PASS" : "FAIL");
    }

    // ---- Thrust reference ----
    float r4;
    thrust::device_ptr<float> tp(d_in);
    float t4 = timeit([&]{ return thrust::reduce(tp,tp+n,0.f); }, r4);
    printf("\nThrust reduce: %.4f ms  (%.2fx vs CPU)  %s\n", t4, cpu_ms/t4, ok(r4) ? "PASS" : "FAIL");

    // ---- Host-to-device transfer cost ----
    float dummy;
    float t_h2d = timeit([&]{ CK(cudaMemcpy(d_in,h.data(),n*sizeof(float),cudaMemcpyHostToDevice)); return 0.f; }, dummy);
    float rv3;
    float t_v3 = timeit([&]{ return run_v3(d_in,d_out,n,256); }, rv3);
    printf("\n== Transfer cost ==\n");
    printf("Host-to-device copy of %.1f MB: %.4f ms (%.1f GB/s)\n", bytes/1e6, t_h2d, bytes/(t_h2d*1e6));
    printf("v3 kernel (block 256): %.4f ms\n", t_v3);
    printf("Copy takes %.1fx longer than the reduction kernel\n", t_h2d/t_v3);
    printf("Copy + v3 total: %.4f ms vs CPU %.4f ms (%.2fx)\n", t_h2d+t_v3, cpu_ms, cpu_ms/(t_h2d+t_v3));

    cudaFree(d_in); cudaFree(d_a); cudaFree(d_b); cudaFree(d_out);
    return 0;
}