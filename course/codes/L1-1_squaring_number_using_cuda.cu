#include <stdio.h>

// __global__ is a CUDA C keyword saying that the function executes on device
// (GPU) calls from the host (CPU) code.
__global__ void square(float* d_out, float* d_in) {
  int idx = threadIdx.x;
  float f = d_in[idx];
  d_out[idx] = f * f;
}

int main(int argc, char** argv) {

  // declare the array size
  const int ARRAY_SIZE = 64;
  const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

  // generate the input array on the host
  float h_in[ARRAY_SIZE];
  for (int i = 0; i < ARRAY_SIZE; i++) {
    h_in[i] = float(i);
  }
  float h_out[ARRAY_SIZE];

  // GPU memory pointer for input and output array
  float* d_in;
  float* d_out;

  // cudaMalloc: allocate GPU memory based on the array size
  // (void**): return information of the CUDA API function
  cudaMalloc((void**) &d_in, ARRAY_BYTES);
  cudaMalloc((void**) &d_out, ARRAY_BYTES);

  // transfer the array to the GPU
  // (destination, source, size, direction)
  // enum cudaMemcpyKind
  // cudaMemcpyHostToDevice
  // cudaMemcpyDeviceToHost
  // cudaMemcpyDeviceToDevice
  cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);

  // launch the kernel
  // <<<numBlocks, numThreadsPerBlock>>>
  // numBlocks: 1, 2, or 3D => dim3(x, y, z)
  // ex: dim3(w, 1, 1) = dim3(w) = w
  //
  // NOTE: Configuring The Kernel Launch
  // KERNEL<<<Grid of blocks, Blocks of Threads>>>
  // ==>
  // KERNEL<<<dim3(bx, by, bz), dim3(tx, ty, tz), shmem>>>
  // 
  // shmem: shared memory per block in bytes
  // threadidx: thread within blocks (threadidx.x, threadidx.y)
  // blockDim: size of a block
  // blockidx: block within grid
  // gridDim; size od grid
  
  square<<<1, ARRAY_SIZE>>>(d_out, d_in);

  // copy back the result array to the CPU
  // (destination, source, size, direction)
  cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);

  // print out result
  for (int i = 0; i < ARRAY_SIZE; i++) {
    printf("%f", h_out[i]);
    printf(((i % 4) != 3) ? "\t" : "\n");
  }

  // free GPU memory allocation
  cudaFree(d_in);
  cudaFree(d_out);

  return 0;
}

// nvcc -x cu L1-1_squaring_number_using_cuda.cu 
