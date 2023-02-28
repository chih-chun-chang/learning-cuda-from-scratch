#include <iostream>
#include <stdio.h>

#define BLOCK_SIZE 128

// Reduction 5: Instruction Overhead
// => Ancillary instructions that are not loads, stores, or arithmetic for the
// core computation
// => unroll loops
__device__ void warpReduce(volatile int* s_data, int tid) { 
  //volatile: tell the compiler not optimize
  s_data[tid] += s_data[tid + 32];
  s_data[tid] += s_data[tid + 16];
  s_data[tid] += s_data[tid + 8];
  s_data[tid] += s_data[tid + 4];
  s_data[tid] += s_data[tid + 2];
  s_data[tid] += s_data[tid + 1];
}

// * As reduction proceeds, # active threads devreases. 
// When s <= 32, only one warp left
// Instruction are SIMD synchronous within a warp
// => when s <= 32:
//    * don't need to syncthreads()
//    * don't need "if (tid < s)"
__global__ void reduce4(int *g_idata, int *g_odata, size_t N) {
  extern __shared__ int s_data[]; //dynamically
  
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

  s_data[tid] = 0;
  
  if (i < N)
    s_data[tid] = g_idata[i] + g_idata[i + blockDim.x];

  __syncthreads();

  // do reduction in shared mem
  for (unsigned int s=blockDim.x/2; s>32; s>>=1) {
    if (tid < s) {
      s_data[tid] += s_data[tid+s];
    }
    __syncthreads();
  }

  // write the result for this block to global mem
  if (tid < 32)
    warpReduce(s_data, tid);
  
  if (tid == 0)
    atomicAdd(&g_odata[0], s_data[0]);
}

void sum_by_cpu(int *input, int *output, size_t N) {
  int sum = 0;
  for (int i = 0; i < N; i++) {
    sum += input[i];
  }
  *output = sum;
}

int main(int argc, char* argv[]) {

  if (argc != 2) {
    std::cerr<< "usage: ./a.out N\n";
    std::exit(EXIT_FAILURE);
  }

  srand(time(NULL));
  size_t N = std::atoi(argv[1]);
  int *input = new int[N];
  int *output = new int[N];

  for (size_t i = 0; i < N; i++) {
    input[i] = rand() % 10;
  }

  // to store the result from host and device
  int cpu_sum, gpu_sum;

  // alocate memory in the device
  int *d_input;
  int *d_output;
  
  cudaMalloc((void **) &d_input, sizeof(int)*N);
  cudaMalloc((void **) &d_output, sizeof(int)*N);

  cudaMemcpy(d_input, input, sizeof(int)*N, cudaMemcpyHostToDevice);

  // timing
  float elapsed_time_gpu, elapsed_time_cpu;
  cudaEvent_t beg, end;
  cudaEventCreate(&beg);
  cudaEventCreate(&end);

  // cpu
  cudaEventRecord(beg, 0);
  sum_by_cpu(input, &cpu_sum, N);
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&elapsed_time_cpu, beg, end);
  printf("CPU time: %.3f ms \n", elapsed_time_cpu);

  // set grid size and block size
  unsigned int block_sz = BLOCK_SIZE;
  unsigned int grid_sz = (N + block_sz - 1) / block_sz;

  // gpu
  cudaEventRecord(beg, 0);
  reduce4<<<grid_sz, block_sz>>>(d_input, d_output, N);
  cudaMemcpy(&gpu_sum, d_output, sizeof(int), cudaMemcpyDeviceToHost);
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&elapsed_time_gpu, beg, end);
  printf("GPU time: %.3f ms \n", elapsed_time_gpu);

  printf("Reduction #4 -> Match: %s \n", cpu_sum==gpu_sum ? "True" : "False");
  printf("%d %d\n", cpu_sum, gpu_sum);

  return 0;  
  
}
