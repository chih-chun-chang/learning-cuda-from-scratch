#include <iostream>
#include <chrono>

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define BLOCK_SIZE 1024

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
    g_odata[blockIdx.x] = s_data[0];
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
  int *d_output_reduce0;
  int *d_output_reduce1;
  int *d_output_reduce2;
  int *d_output_reduce3;
  int *d_output_reduce4;
  int *d_output_reduce5;
  int *d_output_reduce6;
  
  cudaMalloc((void **) &d_input, sizeof(int)*N);
  cudaMalloc((void **) &d_output_reduce0, sizeof(int)*N);
  cudaMalloc((void **) &d_output_reduce1, sizeof(int));
  cudaMalloc((void **) &d_output_reduce2, sizeof(int));
  cudaMalloc((void **) &d_output_reduce3, sizeof(int));
  cudaMalloc((void **) &d_output_reduce4, sizeof(int));
  cudaMalloc((void **) &d_output_reduce5, sizeof(int));
  cudaMalloc((void **) &d_output_reduce6, sizeof(int));

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
  unsigned int max_elems_per_block = block_sz * 2;
  unsigned int grid_sz = 0;
	if (N <= max_elems_per_block) {
		grid_sz = (unsigned int)std::ceil(float(N) / float(max_elems_per_block));
	}
	else
	{
		grid_sz = N / max_elems_per_block;
		if (N % max_elems_per_block != 0)
			grid_sz++;
	}

  printf("%d, %d, %d\n", grid_sz, block_sz, sizeof(unsigned int) *max_elems_per_block);
  dim3 grid_size(1,1,1);
  dim3 block_size(1024,1,1);
  reduce0<<<1, 1024, 8192>>>(d_input, d_output_reduce0);
  //reduce0<<<grid_sz, block_sz, sizeof(unsigned int) * max_elems_per_block>>>(d_input, d_output_reduce0);

  cudaMemcpy(output, d_output_reduce0, sizeof(int)*N, cudaMemcpyDeviceToHost);
  
  printf("Reduction #1() Match: %s \n", cpu_sum==output[0] ? "True" : "False");

  printf("%d %d\n", cpu_sum, output[0]);

  return 0;  
  
}
