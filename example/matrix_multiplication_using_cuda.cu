#include <stdio.h>

#define BLOCK_SIZE 32

// ------------------------------------------------------------------
// &d_A: GPU device pointer to N x K matrix A
// &d_B: GPU device pointer to K x M matrix B
// &d_C: GPU device pointer to N x M matrix C to store the result
// ------------------------------------------------------------------
__global__ void matmul2d_cuda(int* d_A, int* d_B, int* d_C, int N, int K, int
    M) { 
  int row = blockIdx.y * blockDim.y + threadIdx.y; 
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int sum = 0;
  if (col < M && row < N) {
    for(int k = 0; k < K; k++) {
      sum += d_A[row * K + k] * d_B[k * M + col];
    }
    d_C[row * M + col] = sum;
  }
} 

// ------------------------------------------------------------------
// cuda using shared memory: When the matrix dimensions are not multiples 
// of the tile dimensions, then it can happen that some tiles cover the 
// matrices only partially. The tile elements falling outside the not-fully 
// overlapping tiles should be properly zero-ed.
// &d_A: GPU device pointer to N x K matrix A
// &d_B: GPU device pointer to K x M matrix B
// &d_C: GPU device pointer to N x M matrix C to store the result
// ------------------------------------------------------------------
__global__ void matmul2d_cuda_shared_memory(int* d_A, int* d_B, int* d_C, int
    N, int K, int M) {
  // shared tile size = block size
  __shared__ int tile_A[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ int tile_B[BLOCK_SIZE][BLOCK_SIZE];

  int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
  int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  int sum = 0;

  // fill in A and B into tile_A and tile_B
  for (int t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; t++) {
    if(t*BLOCK_SIZE + threadIdx.x < K && row < N) {
      tile_A[threadIdx.y][threadIdx.x] = d_A[row*K + t*BLOCK_SIZE
        + threadIdx.x];
    } else {
      tile_A[threadIdx.y][threadIdx.x] = 0;
    }
    if(t*BLOCK_SIZE + threadIdx.y < K && col < M) {
      tile_B[threadIdx.y][threadIdx.x] = d_B[(t*BLOCK_SIZE + threadIdx.y)*M
        + col];
    } else {
      tile_B[threadIdx.y][threadIdx.x] = 0;
    }
     __syncthreads();

    for (int k = 0; k < BLOCK_SIZE; k++) {
      sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
    }
    __syncthreads();
  }
  if(row < N && col < M) {
    d_C[row * M + col] = sum;
  }
}


// ------------------------------------------------------------------
// &h_A: CPU host pointer to N x K matrix A
// &h_B: CPU host pointer to K x M matrix B
// &h_C: CPU host pointer to N x M matrix C to store the result
// ------------------------------------------------------------------
void matmul_cpu(int* h_A, int* h_B, int* h_C, int N, int K, int M) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            int c = 0;
            for (int k = 0; k < K; k++) {
                c += h_A[i * K + k] * h_B[k * M + j];
            }
            h_C[i * M + j] = c;
        }
    }
}

int main(int argc, char *argv[]) {
  int N, K, M;
  srand(3333);
  printf("Please type in N, K and M:\n");
  printf("A_{NxK} * B_{KxM} = C_{NxM}\n");
  scanf("%d %d %d", &N, &K, &M);

  // calculate size of A, B and C
  const int size_A = N*K;
  const int size_B = K*M;
  const int size_C = N*M;

  // allocate memory in the CPU host RAM
  int* h_A;
  int* h_B;
  int* host_C; // save the result from host CPU
  int* device_C; // save the result from device GPU
  int* device_C_shared;
  cudaMallocHost((void**) &h_A, sizeof(int)*size_A);
  cudaMallocHost((void**) &h_B, sizeof(int)*size_B);
  cudaMallocHost((void**) &host_C, sizeof(int)*size_C);
  cudaMallocHost((void**) &device_C, sizeof(int)*size_C);
  cudaMallocHost((void**) &device_C_shared, sizeof(int)*size_C);

  // random initialize matrix
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < K; j++) {
      h_A[i * M + j] = rand() % 1024;
    }
  }
  for (int i = 0; i < K; i++) {
    for (int j = 0; j < M; j++) {
      h_B[i * M + j] = rand() % 1024;
    }
  }

  // Allocate memory space on the GPU device 
  int* d_A; 
  int* d_B;
  int* d_C;
  int* d_C_shared;
  cudaMalloc((void**) &d_A, sizeof(int)*size_A);
  cudaMalloc((void**) &d_B, sizeof(int)*size_B);
  cudaMalloc((void**) &d_C, sizeof(int)*size_C);
  cudaMalloc((void**) &d_C_shared, sizeof(int)*size_C);  

  // copy matrix A and B from host to device memory
  cudaMemcpy(d_A, h_A, sizeof(int)*size_A, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, sizeof(int)*size_B, cudaMemcpyHostToDevice);
  
  // set grid size and block size
  unsigned int grid_rows = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
  unsigned int grid_cols = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
  
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid(grid_cols, grid_rows);

  //timeing
  float elapsed_time_gpu;
  float elapsed_time_gpu_shared_memory;
  float elapsed_time_cpu;
  cudaEvent_t beg, end;
  cudaEventCreate(&beg);
  cudaEventCreate(&end);

  // --------------------------------------------------------------------
  // --------------------------------------------------------------------
  // GPU Execution
  // --------------------------------------------------------------------
  // --------------------------------------------------------------------
  cudaEventRecord(beg, 0);

  // launch kernel
  matmul2d_cuda<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N, K, M);

  // transfer results back to host
  cudaMemcpy(device_C, d_C, sizeof(int)*size_C, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);

  // compute the gpu elapsed time
  cudaEventElapsedTime(&elapsed_time_gpu, beg, end);
  // --------------------------------------------------------------------
  // --------------------------------------------------------------------
  // GPU Execution with shared memory
  // --------------------------------------------------------------------
  // --------------------------------------------------------------------
  cudaEventRecord(beg, 0);

  // launch kernel
  matmul2d_cuda_shared_memory<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N, K, M);
 
  // transfer results back to host 
  cudaMemcpy(device_C_shared, d_C, sizeof(int)*size_C, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);

  // compute the gpu elapsed time
  cudaEventElapsedTime(&elapsed_time_gpu_shared_memory, beg, end);
  // --------------------------------------------------------------------
  // --------------------------------------------------------------------
  // CPU Execution
  // --------------------------------------------------------------------
  // --------------------------------------------------------------------
  cudaEventRecord(beg, 0);

  // launch cpu-version function
  matmul_cpu(h_A, h_B, host_C, N, K, M);

  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);

  // compute the cpu elapsed time
  cudaEventElapsedTime(&elapsed_time_cpu, beg, end);
  // --------------------------------------------------------------------
  // --------------------------------------------------------------------


  // check
  bool check = true;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      if (host_C[i*M + j] != device_C[i*M + j] || host_C[i*M + j] !=
          device_C_shared[i*M + j]) {
        check = false;
        break;
      }
    }
  }

  printf("A_{%dx%d} * B_{%dx%d} = C_{%dx%d}\n", N, K, K, M, N, M);
  if (check) {
    printf("Correct!\n");
    printf("GPU elapsed time (w/o shared memory) %.3f ms\n", elapsed_time_gpu);
    printf("GPU elapsed time (with shared memory) %.3f ms\n",
        elapsed_time_gpu_shared_memory);
    printf("CPU elapsed time %.3f ms\n", elapsed_time_cpu);
    printf("GPU speed up = %.3fx\n",
        elapsed_time_cpu/elapsed_time_gpu_shared_memory);
  } else {
    printf("Results wrong!\n");
  }

  // free memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaFreeHost(h_A);
  cudaFreeHost(h_B);
  cudaFreeHost(host_C);
  cudaFreeHost(device_C);
  cudaFreeHost(device_C_shared);
  return 0;
}
