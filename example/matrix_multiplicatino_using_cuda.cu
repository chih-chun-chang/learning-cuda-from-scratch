#include <stdio.h>

#define N 50
#define K 40
#define M 30

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

int main(int argc, char**) {

  // matrix size N*M
  const int MAT_C_SIZE = N * M * sizeof(float);
  const int MAT_A_SIZE = N * K * sizeof(float);
  const int MAT_B_SIZE = K * M * sizeof(float);
  
  // generate matrices and print
  float h_A[N*K];
  float h_B[K*M];
  float h_C[N*M];

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
  matmul2d<<<numsBlocks, threadsPerBlock>>>(d_C, d_A, d_B);

  // Date Transfer (device -> host)
  cudaMemcpy(h_C, d_C, MAT_C_SIZE, cudaMemcpyDeviceToHost);

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
  printf("Matrix C: \n");
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      printf("%f\t", h_C[i*M + j]);
    }   
    printf("\n");
  }

  return 0;
}
  
