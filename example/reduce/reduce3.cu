#include <iostream>
#include <stdio.h>

#define BLOCK_SIZE 128

// Reduction #4: Sequential Addressing (First Add During Load)
__global__ void reduce3(int *g_idata, int *g_odata, size_t N) {
  extern __shared__ int s_data[]; //dynamically
  
  // perform first level of reduction
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
  
  s_data[tid] = 0; 
  
  if (i < N)
    s_data[tid] = g_idata[i] + g_idata[i + blockDim.x];
  
  __syncthreads();


  // do reduction in shared mem
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
  if (tid == 0){
    atomicAdd(&g_odata[0], s_data[0]);

  }
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
  //printf("grid sz %d\n", grid_sz);
  reduce3<<<grid_sz, block_sz>>>(d_input, d_output, N);
  cudaMemcpy(&gpu_sum, d_output, sizeof(int), cudaMemcpyDeviceToHost);
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&elapsed_time_gpu, beg, end);
  printf("GPU time: %.3f ms \n", elapsed_time_gpu);

  printf("Reduction #3 -> Match: %s \n", cpu_sum==gpu_sum ? "True" : "False");
  //printf("%d %d\n", cpu_sum, gpu_sum);

  return 0;  
  
}
