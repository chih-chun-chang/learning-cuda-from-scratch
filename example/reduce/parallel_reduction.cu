#include "cuda_runtime.h"

#include "device_launch_parameters.h"

/*********************************************
 * ref: https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
 * ******************************************/

// Avoid global sync by decomposing computation into multiple kernel invoations


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

/*
// Reduction #2: Interleaved Addressing (stride index and non-divergent branch)
// --------------------------------------------
// Problem: Shared Memory Bank Conflicts
// Shared memory is divided into equally sized memory banks that can be
// serviced simultaneously.
// If multiple threads' requested address map to the same memory bank, the
// accesses are serialized. 
// --------------------------------------------
__global__ void reduce1(int *g_idata, int *g_odata) {
  extern __shared__ int s_data[]; //dynamically
  
  // each thread loads one element from global to shared mem
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  s_data[tid] = g_idata[i];
  __syncthreads();

  // do reduction in shared mem
  for (unsigned int s=1; s < blockDim.x; s*=2) {
    int index = 2*s*tid;
    if (index < blockDim.x) {
      s_data[index] += s_data[index+s];
    }
    syncthreads();
  }

  // write the result for this block to global mem
  if (tid == 0)
    g_odata[blockIdx.x] = s_data[0];
}


// Reduction #3: Sequential Addressing (sequential addressing is conflict free)
__global__ void reduce2(int *g_idata, int *g_odata) {
  extern __shared__ int s_data[]; //dynamically
  
  // each thread loads one element from global to shared mem
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  s_data[tid] = g_idata[i];
  __syncthreads();

  // do reduction in shared mem
  // 1st round: (stride = 8)
  //    tid = 0(+8), 1(+8), ...
  // 2nd round: (stride = 4)
  //    tid = 0(+4), 1(+4), ...
  // 3rd round: (stride = 2)
  //    tid = 0(+2), 1(+2)
  // 4nd round: (stride = 1)
  //    tid = 0(+1)
  for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
    // ----------------------------------------
    // Problem Here:
    // Half of the threads are idle on first loop iteration!
    // ----------------------------------------
    if (tid < s) {
      s_data[tid] += s_data[tid+s];
    }
    syncthreads();
  }

  // write the result for this block to global mem
  if (tid == 0)
    g_odata[blockIdx.x] = s_data[0];
}


// Reduction #4: Sequential Addressing (First Add During Load)
__global__ void reduce3(int *g_idata, int *g_odata) {
  extern __shared__ int s_data[]; //dynamically
  
  // perform first level of reduction
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * (blockDim.x*2) + threadIdx.x;
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
    syncthreads();
  }

  // write the result for this block to global mem
  if (tid == 0)
    g_odata[blockIdx.x] = s_data[0];
}

// Reduction 5: Instruction Overhead
// => Ancillary instructions that are not loads, stores, or arithmetic for the
// core computation
// => unroll loops
__device__ void warpReduce(volatile int* s_data, int tid) { 
  //volatile: tell the compiler not optimize
  s_data[tid] += s_data[tid + 32];
  s_data[tid] += s_data[tid + 16];
  s_data[tid] += s_data[tid + 8];
  s_data[tid] += s_data[tid + 4];
  s_data[tid] += s_data[tid + 2];
  s_data[tid] += s_data[tid + 1];
}t
// * As reduction proceeds, # active threads devreases. 
// When s <= 32, only one warp left
// Instruction are SIMD synchronous within a warp
// => when s <= 32:
//    * don't need to syncthreads()
//    * don't need "if (tid < s)"
__global__ void reduce4(int *g_idata, int *g_odata) {
  extern __shared__ int s_data[]; //dynamically
  
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * (blockDim.x*2) + threadIdx.x;
  s_data[tid] = g_idata[i] + g_idata[i + blockDim.x];
  __syncthreads();

  // do reduction in shared mem
  for (unsigned int s=blockDim.x/2; s>32; s>>=1) {
    if (tid < s) {
      s_data[tid] += s_data[tid+s];
    }
    syncthreads();
  }

  // write the result for this block to global mem
  if (tid < 32)
    warpReduce(s_data, tid);
}

// Reduction 6: Complete Unrolling
// How can we unroll for block size that we don't know at compile time? => CUDA
// supports C++ template  parameters on device and host functions
template <unsigned int blockSize>
__device__ void warpReduce(volatile int* s_data, int tid) { 
  //volatile: tell the compiler not optimize
  if (blockSize >=64)
    s_data[tid] += s_data[tid + 32];
  if (blockSize >=32)
    s_data[tid] += s_data[tid + 16];
  if (blockSize >=16)
    s_data[tid] += s_data[tid + 8];
  if (blockSize >=8)
    s_data[tid] += s_data[tid + 4];
  if (blockSize >=4)
    s_data[tid] += s_data[tid + 2];
  if (blockSize >=2)
    s_data[tid] += s_data[tid + 1];

template <unsigned int blockSize>
__global__ void reduce5(int *g_idata, int *g_odata    ) {
  extern __shared__ int s_data[]; //dynamically
  
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * (blockDim.x*2) + threadIdx.x;
  s_data[tid] = g_idata[i] + g_idata[i + blockDim.x];
  __syncthreads();

  if (blockSize >=512) {
    if (tid < 256) {
      s_data[tid] += s_data[tid + 256];
    }
    __syncthreads();
  }
  if (blockSize >=256) {
    if (tid < 128) {
      s_data[tid] += s_data[tid + 128];
    }
    __syncthreads();
  }
  if (blockSize >=128) {
    if (tid < 64) {
      s_data[tid] += s_data[tid + 64];
    }
    __syncthreads();
  }

  if (tid < 32)
    warpReduce<blockSize>(s_data, tid);
  if (tid == 0)
    g_odata[blockIdx.x] = s_data[0];
}

// Reduction 7: Multiple Adds / Thread
template <unsigned int blockSize>
__global__ void reduce6(int *g_idata, int *g_odata, unsigned int n) {
  extern __shared__ int s_data[];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*(blockSize*2) + tid;
  unsigned int gridSize = blockSize * 2 * gridDim.x;
  s_data[tid] = 0;

  while (i < n) {
    s_data[tid] += g_idata[i] + g_idata[i + blockSize];
    i += gridSize;
  }
  __syncthreads();

  if (blockSize >=512) {
    if (tid < 256) {
      s_data[tid] += s_data[tid + 256];
    }
    __syncthreads();
  }
  if (blockSize >=256) {
    if (tid < 128) {
      s_data[tid] += s_data[tid + 128];
    }
    __syncthreads();
  }
  if (blockSize >=128) {
    if (tid < 64) {
      s_data[tid] += s_data[tid + 64];
    }
    __syncthreads();
  }

  if (tid < 32)
    warpReduce<blockSize>(s_data, tid);
  if (tid == 0)
    g_odata[blockIdx.x] = s_data[0];
}
*/


/*
 switch (threads) {
case 512:
reduce5<512><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata); break;
case 256:
reduce5<256><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata); break;
case 128:
reduce5<128><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata); break;
case 64:
reduce5< 64><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata); break;
case 32:
reduce5< 32><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata); break;
case 16:
reduce5< 16><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata); break;
case 8:
reduce5<
case 4:
reduce5<
case 2:
reduce5<
case 1:
reduce5<
8><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata); break; 4><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata); break; 2><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata); break; 1><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata); break;
}
 * */


