#include <iostream>
#include <stdio.h>

#define BLOCK_SIZE 128

// Reduction #3: Sequential Addressing (sequential addressing is conflict free)
__global__ void reduce2(int *g_idata, int *g_odata) {
  extern __shared__ int s_data[]; //dynamically

  // each thread loads one element from global to shared mem
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  s_data[tid] = g_idata[i];
  __syncthreads();

  // do reduction in shared mem
  // 1st round: (stride = 8)
  //    tid = 0(+8), 1(+8), ...
  // 2nd round: (stride = 4)
  //    tid = 0(+4), 1(+4), ...
  // 3rd round: (stride = 2)
  //    tid = 0(+2), 1(+2)
  // 4nd round: (stride = 1)
  //    tid = 0(+1)
  for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
    // ----------------------------------------
    // Problem Here:
    // Half of the threads are idle on first loop iteration!
    // ----------------------------------------
    if (tid < s) {
      s_data[tid] += s_data[tid+s];
    }
    __syncthreads();
  }

  // write the result for this block to global mem
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
  reduce2<<<grid_sz, block_sz>>>(d_input, d_output);
  cudaMemcpy(&gpu_sum, d_output, sizeof(int), cudaMemcpyDeviceToHost);
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&elapsed_time_gpu, beg, end);
  printf("GPU time: %.3f ms \n", elapsed_time_gpu);

  printf("Reduction #2 -> Match: %s \n", cpu_sum==gpu_sum ? "True" : "False");
  //printf("%d %d\n", cpu_sum, output[0]);

  return 0;  
  
}
