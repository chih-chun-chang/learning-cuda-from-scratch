#include <stdio.h>

#define N 15
#define K 14
#define M 13

#define BLOCK_SIZE 16

// C: N*M
// A: N*K
// B: K*M
__global__ void matmul2d(float* d_C, float* d_A, float* d_B) {
  int rows = blockIdx.x * blockDim.x + threadIdx.x;
  int cols = blockIdx.y * blockDim.y + threadIdx.y;

  if (rows < N && cols < M) {
    float c = 0;
    for (int k = 0; k < K; k++) {
      c += d_A[rows*K + k] * d_B[M*k + cols]; // A[n][k] * B[k][m]
    }
    d_C[rows*M + cols] = c;
  }
}

// C: N*M
// A: N*K
// B: K*M
// 2D Matrix Multiplication Using Shared Memory
// Shared Memory: shared by all threads in a thread block
__global__ void matmul2d_shared_memory(float* d_C, float* d_A, float* d_B) {
  //create shared memory of matrix tiles
  __shared__ float tile_A[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float tile_B[BLOCK_SIZE][BLOCK_SIZE];
  
  int rows = blockIdx.x * blockDim.x + threadIdx.x;
  int cols = blockIdx.y * blockDim.y + threadIdx.y;
  float c = 0;
  
  // read matrix tile into shared memory
  int numTiles = ((K - 1) / BLOCK_SIZE) + 1;
  for (int t = 0; t < numTiles; t++) {
    
    // load d_A to tile_A
    if (rows < N && (threadIdx.y + t*BLOCK_SIZE) < K) {
      tile_A[threadIdx.x][threadIdx.y] = d_A[rows*K + threadIdx.y
        + t*BLOCK_SIZE];
    } else {
      tile_A[threadIdx.x][threadIdx.y] = 0;
    }
    
    // load d_B to tile_B
    if (cols < M && (threadIdx.x + t*BLOCK_SIZE) < K) {
      tile_B[threadIdx.x][threadIdx.y] = d_B[(threadIdx.x + t*BLOCK_SIZE)*M
        + cols];
    } else {
      tile_B[threadIdx.x][threadIdx.y] = 0;
    }
    
    // make sure all reads have completed
    __syncthreads();


    // mul
    for (int k = 0; k < BLOCK_SIZE; k++) {
      c += tile_A[threadIdx.x][k] * tile_B[k][threadIdx.y];
    }
  }
  
  // save to output
  if (rows < N && cols < M) {
    d_C[rows*M + cols] = c;
  }
}

// C: N*M
// A: N*K
// B: K*M
void matmul2d_cpu(float* h_C, float* h_A, float* h_B) {
  for (int i = 0; i < N; i++) {
    for(int j = 0; j < M; j++) {
      h_C[i*M + j] = 0;
      for (int k = 0; k < K; k++) {
        h_C[i*M + j] += h_A[i*K + k] * h_B[k*M + j];
      }
    }
  }
  return;
}

int main(int argc, char**) {

  // matrix size N*M
  const int MAT_C_SIZE = N * M * sizeof(float);
  const int MAT_A_SIZE = N * K * sizeof(float);
  const int MAT_B_SIZE = K * M * sizeof(float);
  
  // generate matrices and print
  float h_A[N*K];
  float h_B[K*M];
  float host_C[N*M];
  float device_C[N*M];

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < K; j++) {
      h_A[i*K + j] = float(i*K + j); 
    }   
  }
  for (int i = 0; i < K; i++) {
    for (int j = 0; j < M; j++) {
      h_B[i*M + j] = float(i*M + j); 
    }   
  }
  
  // GPU memory pointer
  float* d_A;
  float* d_B;
  float* d_C;
  
  // GPU memory allocation
  cudaMalloc((void**) &d_A, MAT_A_SIZE);
  cudaMalloc((void**) &d_B, MAT_B_SIZE);
  cudaMalloc((void**) &d_C, MAT_C_SIZE);

  // Data Transfer (host -> device)
  cudaMemcpy(d_A, h_A, MAT_A_SIZE, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, MAT_B_SIZE, cudaMemcpyHostToDevice);

  // Launch Kernel
  unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
  unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
  dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 numsBlocks(grid_cols, grid_rows);
  //matmul2d<<<numsBlocks, threadsPerBlock>>>(d_C, d_A, d_B);
  matmul2d_shared_memory<<<numsBlocks, threadsPerBlock>>>(d_C, d_A, d_B);

  // Date Transfer (device -> host)
  cudaMemcpy(device_C, d_C, MAT_C_SIZE, cudaMemcpyDeviceToHost);


  // CPU matmul
  matmul2d_cpu(host_C, h_A, h_B);

  // print output
  
  printf("Matrix A: \n");
  for (int i = 0; i < N; i++) {
    for (int k = 0; k < K; k++) {
      printf("%f\t", h_A[i*K + k]);
    }   
    printf("\n");
  }
  printf("Matrix B: \n");
  for (int k = 0; k < K; k++) {
    for (int j = 0; j < M; j++) {
      printf("%f\t", h_B[k*M + j]);
    }   
    printf("\n");
  }
  printf("Matrix C(device): \n");
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      printf("%f\t", device_C[i*M + j]);
    }   
    printf("\n");
  }
  printf("Matrix C(host): \n");
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      printf("%f\t", host_C[i*M + j]);
    }
    printf("\n");
  }
  

  
  // compare matrix if equal
  for (int i = 0; i < M*N; i++){
    if (host_C[i] != device_C[i]) {
      printf("[%d][%d]: Host = %f \t ; Device = %f \n", i/M, i%M,
          host_C[i], device_C[i]);
      break;
    }
  }
  
  // Free
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  return 0;
}
  
