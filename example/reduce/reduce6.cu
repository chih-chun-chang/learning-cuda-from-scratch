#include <iostream>
#include <stdio.h>

#define BLOCK_SIZE 128

template <unsigned int blockSize>
__device__ void warpReduce(volatile int *s_data, unsigned int tid) {
  if (blockSize >= 64) s_data[tid] += s_data[tid + 32];
  if (blockSize >= 32) s_data[tid] += s_data[tid + 16]; 
  if (blockSize >= 16) s_data[tid] += s_data[tid + 8]; 
  if (blockSize >= 8) s_data[tid] += s_data[tid + 4]; 
  if (blockSize >= 4) s_data[tid] += s_data[tid + 2]; 
  if (blockSize >= 2) s_data[tid] += s_data[tid + 1];
}

template <unsigned int blockSize>
__global__ void reduce6(int *g_idata, int *g_odata, size_t N) {
  extern __shared__ int s_data[];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * (blockSize * 2) + tid; 
  unsigned int gridSize = blockSize * 2 * gridDim.x; 
  
  s_data[tid] = 0;

  while (i < N) { 
    s_data[tid] += g_idata[i] + g_idata[i+blockSize]; 
    i += gridSize; 
  } 
  __syncthreads();

  if (blockSize >= 512) { 
    if (tid < 256) { 
      s_data[tid] += s_data[tid + 256]; 
    } 
    __syncthreads(); 
  } 
  
  if (blockSize >= 256) { 
    if (tid < 128) { 
      s_data[tid] += s_data[tid + 128]; 
    }
    __syncthreads(); 
  } 
  
  if (blockSize >= 128) { 
    if (tid < 64) { 
      s_data[tid] += s_data[tid + 64]; 
    } 
    __syncthreads(); 
  }
  
  if (tid < 32) 
    warpReduce<blockSize>(s_data, tid);

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
  reduce6<BLOCK_SIZE><<<grid_sz, block_sz>>>(d_input, d_output, N);
  cudaMemcpy(&gpu_sum, d_output, sizeof(int), cudaMemcpyDeviceToHost);
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&elapsed_time_gpu, beg, end);
  printf("GPU time: %.3f ms \n", elapsed_time_gpu);

  printf("Reduction #6 -> Match: %s \n", cpu_sum==gpu_sum ? "True" : "False");
  //printf("%d %d\n", cpu_sum, output[0]);

  return 0;  
  
}
