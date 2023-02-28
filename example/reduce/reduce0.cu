#include <iostream>
#include <stdio.h>

#define BLOCK_SIZE 128

// Reduction #1: Interleaved Addressing
__global__ void reduce0(int *g_idata, int *g_odata) {
  extern __shared__ int s_data[]; //dynamically

  // each thread loads one element from global to shared mem
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  s_data[tid] = g_idata[i];
  
  __syncthreads();

  // do reduction in shared mem
  // 1st round: (stride = 1)
  //    tid = 0, 2, 4, 6, 8
  // 2nd round: (stride = 2)
  //    tid = 0, 4, 8
  // 3rd round: (stride = 4)
  //    tid = 0, 8
  // 4nd round: (stride = 8)
  //    tid = 0
  for (unsigned int s=1; s < blockDim.x; s*=2) {
    if (tid % (2*s) == 0) {
      // ------------------------------------
      // Problem Here!!
      // * Highly divergent
      // => warps are very inefficient
      // * % operator is very slow
      // ------------------------------------
      s_data[tid] += s_data[tid+s];
    }
    __syncthreads();
  }

  // write the result for this block to global mem
  if (tid == 0)
    atomicAdd(&g_odata[0], s_data[0]);
    //g_odata[0] += s_data[0];
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
  reduce0<<<grid_sz, block_sz>>>(d_input, d_output);
  cudaMemcpy(&gpu_sum, d_output, sizeof(int), cudaMemcpyDeviceToHost);
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&elapsed_time_gpu, beg, end);

  printf("GPU time: %.3f ms \n", elapsed_time_gpu); 

  printf("Reduction #0() Match: %s \n", cpu_sum==gpu_sum ? "True" : "False");

  return 0;  
  
}
