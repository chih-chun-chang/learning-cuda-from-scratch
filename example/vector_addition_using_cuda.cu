#include <stdio.h>


// kernel of vector addition: C = A + B
__global__ void vector_addition(float* d_C, float* d_A, float* d_B) {
  int idx = threadIdx.x;
  float a = d_A[idx];
  float b = d_B[idx];
  d_C[idx] = a + b;
}


int main(int argc, char** argv) {

  const int VECTOR_LENGTH = 128;
  const int VECTOR_BYTES = VECTOR_LENGTH * sizeof(float);

  // generate two vectors A, B
  float h_A[VECTOR_LENGTH];
  float h_B[VECTOR_LENGTH];
  float h_C[VECTOR_LENGTH];
  for (int i = 0; i < VECTOR_LENGTH; i++) {
    h_A[i] = float(i);
    h_B[i] = float(VECTOR_LENGTH + i);
  } 

  // GPU memory pointer for A, B, C
  float* d_A;
  float* d_B;
  float* d_C;

  // allocate GPU memory
  cudaMalloc((void**) &d_A, VECTOR_BYTES);
  cudaMalloc((void**) &d_B, VECTOR_BYTES);
  cudaMalloc((void**) &d_C, VECTOR_BYTES);

  // copy data to GPU
  cudaMemcpy(d_A, h_A, VECTOR_BYTES, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, VECTOR_BYTES, cudaMemcpyHostToDevice);

  // launch kernel
  vector_addition<<<1, VECTOR_LENGTH>>>(d_C, d_B, d_A);

  // copy result back to host
  cudaMemcpy(h_C, d_C, VECTOR_BYTES, cudaMemcpyDeviceToHost);

  // print the result
  printf("A = \n");
  for (int i = 0; i < VECTOR_LENGTH; i++) {
    printf("%f", h_A[i]);
    printf(((i % 4) != 3) ? "\t" : "\n");
  }
  printf("B = \n");
  for (int i = 0; i < VECTOR_LENGTH; i++) {
    printf("%f", h_B[i]);
    printf(((i % 4) != 3) ? "\t" : "\n");
  }
  printf("C = \n");
  for (int i = 0; i < VECTOR_LENGTH; i++) {
    printf("%f", h_C[i]);
    printf(((i % 4) != 3) ? "\t" : "\n");
  }
  
  // free GPU memory allocation
  cudaFree(d_A);
  cudaFree(d_B); 
  cudaFree(d_C);
  
  return 0;
}
