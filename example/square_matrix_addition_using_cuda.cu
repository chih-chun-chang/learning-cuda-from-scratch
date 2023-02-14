#include <stdio.h>

#define N 128 // matrix size
#define BLOCK_SIZE 32


// GPU memory dynamically allocated
// Unable to use two-dimensional indices
// How to accessing matrices in linear memory?
// http://users.wfu.edu/choss/CUDA/docs/Lecture%205.pdf
//
// int col = blockIdx.x * blockDim.x + threadIdx.x;
// int row = blockIdx.y * blockDim.y + threadIdx.y;
// int index = row * N + column;
// A[index] = ...

// Square Matrices Addition: C = A + B
__global__ void matrix_addition(float* d_C, float* d_A, float* d_B) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int index = i*N + j;
  d_C[index] = d_A[index] + d_B[index];
}

int main(int argc, char** argv) {

  const int MATRIX_BYTE = N * N * sizeof(float);

  float h_A[N*N];
  float h_B[N*N];
  float h_C[N*N];

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      h_A[i*N+j] = float(i*N + j);
      h_B[i*N+j] = float(i*N + j);
    }
  }

  // GPU memory pointer
  float* d_A;
  float* d_B;
  float* d_C;

  // Allocate GPU memory
  cudaMalloc((void**) &d_A, MATRIX_BYTE);
  cudaMalloc((void**) &d_B, MATRIX_BYTE);
  cudaMalloc((void**) &d_C, MATRIX_BYTE);

  // Data Transfer
  cudaMemcpy(d_A, h_A, MATRIX_BYTE, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, MATRIX_BYTE, cudaMemcpyHostToDevice);
  
  // Launch Kernel
  unsigned int grid_rows = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
  unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
  
  dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 numBlocks(grid_rows, grid_cols);
  matrix_addition<<<numBlocks, threadsPerBlock>>>(d_C, d_A, d_B);

  // Transfer the result back
  cudaMemcpy(h_C, d_C, MATRIX_BYTE, cudaMemcpyDeviceToHost);
  
  // Print
  printf("A=\n");
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      printf("%f", h_A[i*N+j]);
      printf((((i*N+j) % 4) != 3) ? "\t" : "\n");
    }
  }
  printf("B=\n");
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      printf("%f", h_B[i*N+j]);
      printf((((i*N+j) % 4) != 3) ? "\t" : "\n");
    }
  }
  printf("C=\n");
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      printf("%f", h_C[i*N+j]);
      printf((((i*N+j) % 4) != 3) ? "\t" : "\n");
    }
  }

  // Free
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  return 0;
}
