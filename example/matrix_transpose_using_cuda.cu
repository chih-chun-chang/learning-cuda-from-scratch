#include <stdio.h>

#define N 20 //rows
#define M 30 //cols
#define BLOCK_SIZE 16

// matrix transpose: 
__global__ void matrix_transpose(float* d_mat_out, float* d_mat_in) {
  int rows = blockIdx.x * blockDim.x + threadIdx.x;
  int cols = blockIdx.y * blockDim.y + threadIdx.y;
  if (rows < N && cols < M) {
    int pos = rows * M + cols;
    int trans_pos = cols * N + rows;
    d_mat_out[trans_pos] = d_mat_in[pos];
  }
}

// TODO: Matrix Transpose using shared memory

int main(int argc, char**) {

  // matrix size N*M
  const int MATRIX_SIZE = N * M * sizeof(float);

  // generate matrices and print
  float h_mat_in[N*M];
  float h_mat_out[M*N];
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      h_mat_in[i*M + j] = float(i*M + j);
    }
  }

  // GPU memory pointer
  float* d_mat_in;
  float* d_mat_out;
  
  // GPU memory allocation
  cudaMalloc((void**) &d_mat_in, MATRIX_SIZE);
  cudaMalloc((void**) &d_mat_out, MATRIX_SIZE);

  // Data Transfer (host -> device)
  cudaMemcpy(d_mat_in, h_mat_in, MATRIX_SIZE, cudaMemcpyHostToDevice);

  // Launch Kernel
  unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
  unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
  dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 numsBlocks(grid_cols, grid_rows);
  matrix_transpose<<<numsBlocks, threadsPerBlock>>>(d_mat_out, d_mat_in);

  // Date Transfer (device -> host)
  cudaMemcpy(h_mat_out, d_mat_out, MATRIX_SIZE, cudaMemcpyDeviceToHost);

  // print output
  printf("Matrix: \n");
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      printf("%f\t", h_mat_in[i*M + j]);
    }
    printf("\n");
  }
  printf("Matrix Transpose: \n");
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      printf("%f\t", h_mat_out[i*N + j]);
    }
    printf("\n");
  }

  // Free
  cudaFree(d_mat_in);
  cudaFree(d_mat_out);

  return 0;
}
