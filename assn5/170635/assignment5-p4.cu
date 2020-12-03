// Compile: nvcc -g -G -arch=sm_61 -std=c++11 assignment5-p4.cu -o assignment5-p4

#include <cmath>
#include <cuda.h>
#include <iostream>
#include <sys/time.h>

#define N (1 << 12)
#define THRESHOLD (0.000001)
#define BLOCK_SIZE 32

using std::cerr;
using std::cout;
using std::endl;

__global__ void kernel1(uint64_t* d_A, uint64_t* d_B, uint64_t* d_C) {
  // TODO: Fill in
  uint64_t i = blockIdx.y * blockDim.y + threadIdx.y;
  uint64_t j = blockIdx.x * blockDim.x + threadIdx.x;

  uint64_t val = 0;
  for(uint64_t k = 0; k < N; k++){
    val += d_A[i*N + k] * d_B[k*N + j];
  }
  d_C[i*N + j] = val;
}

__global__ void kernel2(uint64_t* d_A, uint64_t* d_B, uint64_t* d_C) {
  // TODO: Fill in
  uint64_t i = blockIdx.y * blockDim.y + threadIdx.y;
  uint64_t j = blockIdx.x * blockDim.x + threadIdx.x;

  uint64_t top_left_i = blockIdx.y * BLOCK_SIZE;
  uint64_t top_left_j = blockIdx.x * BLOCK_SIZE;

  uint64_t val = 0;

  for(uint64_t block_num = 0; block_num < N/BLOCK_SIZE; block_num++){
    __shared__ uint64_t mat1[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ uint64_t mat2[BLOCK_SIZE][BLOCK_SIZE];

    mat1[threadIdx.y][threadIdx.x] = d_A[(top_left_i + threadIdx.y)*N + (block_num * BLOCK_SIZE + threadIdx.x)];
    mat2[threadIdx.y][threadIdx.x] = d_B[(block_num * BLOCK_SIZE + threadIdx.y)*N + (top_left_j + threadIdx.x)];

    __syncthreads();

    for(uint64_t k = 0; k < BLOCK_SIZE; k++){
      val += mat1[threadIdx.y][k] * mat2[k][threadIdx.x];
    }

    __syncthreads();
    
  }

  d_C[i*N + j] = val;
}

__global__ void kernel3(uint64_t* d_A, uint64_t* d_B, uint64_t* d_C) {
  // TODO: Fill in
  uint64_t i = blockIdx.y * blockDim.y + threadIdx.y;
  uint64_t j = blockIdx.x * blockDim.x + threadIdx.x;

  uint64_t top_left_i = blockIdx.y * BLOCK_SIZE;
  uint64_t top_left_j = blockIdx.x * BLOCK_SIZE;

  uint64_t val = 0;

  for(uint64_t block_num = 0; block_num < N/BLOCK_SIZE; block_num++){
    __shared__ uint64_t mat1[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ uint64_t mat2[BLOCK_SIZE][BLOCK_SIZE];

    mat1[threadIdx.y][threadIdx.x] = d_A[(top_left_i + threadIdx.y)*N + (block_num * BLOCK_SIZE + threadIdx.x)];
    mat2[threadIdx.y][threadIdx.x] = d_B[(block_num * BLOCK_SIZE + threadIdx.y)*N + (top_left_j + threadIdx.x)];

    __syncthreads();

    for(uint64_t k = 0; k < BLOCK_SIZE; k += 4){
      val += mat1[threadIdx.y][k] * mat2[k][threadIdx.x];
      val += mat1[threadIdx.y][k+1] * mat2[k+1][threadIdx.x];
      val += mat1[threadIdx.y][k+2] * mat2[k+2][threadIdx.x];
      val += mat1[threadIdx.y][k+3] * mat2[k+3][threadIdx.x];
    }

    __syncthreads();
    
  }

  d_C[i*N + j] = val;
}

__host__ void cpumatMul(uint64_t* h_A, uint64_t* h_B, uint64_t* h_C) {
  for (uint64_t i = 0; i < N; i++) {
    for (uint64_t j = 0; j < N; j++) {
      uint64_t sum = 0;
      for (uint64_t k = 0; k < N; k++) {
        sum += h_A[i * N + k] * h_B[k * N + j];
      }
      h_C[i * N + j] = sum;
    }
  }
}

__host__ void check_result(uint64_t* w_ref, uint64_t* w_opt) {
  bool wrong = false;
  for (uint64_t i = 0; i < N; i++) {
    for (uint64_t j = 0; j < N; j++) {
      if (w_ref[i * N + j] != w_opt[i * N + j]) {
        wrong = true;
        goto out;
      }
    }
  }
out:
  if (wrong) {
    cout << " Diffs found!" << endl;
  } else {
    cout << "No differences found between base and test versions\n";
  }
}

double rtclock() { // Seconds
  struct timezone Tzp;
  struct timeval Tp;
  int stat;
  stat = gettimeofday(&Tp, &Tzp);
  if (stat != 0) {
    cout << "Error return from gettimeofday: " << stat << "\n";
  }
  return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

int main() {
  uint64_t SIZE = N * N;

  uint64_t *h_A, *h_B, *h_cpu_C, *h_gpu1_C, *h_gpu2_C;

  h_A = (uint64_t*)malloc(SIZE * sizeof(uint64_t));
  h_B = (uint64_t*)malloc(SIZE * sizeof(uint64_t));
  h_cpu_C = (uint64_t*)malloc(SIZE * sizeof(uint64_t));
  h_gpu1_C = (uint64_t*)malloc(SIZE * sizeof(uint64_t));
  h_gpu2_C = (uint64_t*)malloc(SIZE * sizeof(uint64_t));

  for (uint64_t i = 0; i < N; i++) {
    for (uint64_t j = 0; j < N; j++) {
      h_A[i * N + j] = rand() % 64;
      h_B[i * N + j] = 2;
      h_cpu_C[i * N + j] = 0;
      h_gpu1_C[i * N + j] = 0;
      h_gpu2_C[i * N + j] = 0;
    }
  }

  double clkbegin = rtclock();
  cpumatMul(h_A, h_B, h_cpu_C);
  double clkend = rtclock();
  double cpu_time = clkend - clkbegin;
  cout << "Matmul time on CPU: " << cpu_time * 1000 << " msec" << endl;

  cudaError_t status;
  cudaEvent_t start, end;

  uint64_t *d_A, *d_B, *d_C1;
  status = cudaMalloc(&d_A, SIZE * sizeof(uint64_t));
  if (status != cudaSuccess) {
    cerr << cudaGetErrorString(status) << endl;
  }
  status = cudaMalloc(&d_B, SIZE * sizeof(uint64_t));
  status = cudaMalloc(&d_C1, SIZE * sizeof(uint64_t));

  dim3 threadsPerBlock(32,32);
  dim3 numBlocks(N/threadsPerBlock.x, N/threadsPerBlock.y);

  cudaEventCreate(&start);
  cudaEventCreate(&end);

  cudaEventRecord(start, 0);

  status = cudaMemcpy(d_A, h_A, SIZE * sizeof(uint64_t), cudaMemcpyHostToDevice);
  status = cudaMemcpy(d_B, h_B, SIZE * sizeof(uint64_t), cudaMemcpyHostToDevice);

  // TODO: Fill in
  kernel1<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C1);

  cudaMemcpy(h_gpu1_C, d_C1, SIZE * sizeof(uint64_t), cudaMemcpyDeviceToHost);
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);

  float kernel_time;
  cudaEventElapsedTime(&kernel_time, start, end);
  cudaEventDestroy(start);
  cudaEventDestroy(end);

  check_result(h_cpu_C, h_gpu1_C);
  std::cout << "Kernel 1 time (ms): " << kernel_time << "\n";

  // kernel 2
  uint64_t* d_C2;
  threadsPerBlock = dim3(BLOCK_SIZE,BLOCK_SIZE);
  numBlocks = dim3(N/threadsPerBlock.x, N/threadsPerBlock.y);
  status = cudaMalloc(&d_C2, SIZE * sizeof(uint64_t));
  // TODO: Fill in

  cudaEventCreate(&start);
  cudaEventCreate(&end);

  cudaEventRecord(start, 0);

  status = cudaMemcpy(d_A, h_A, SIZE * sizeof(uint64_t), cudaMemcpyHostToDevice);
  status = cudaMemcpy(d_B, h_B, SIZE * sizeof(uint64_t), cudaMemcpyHostToDevice);

  kernel2<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C2);

  cudaMemcpy(h_gpu2_C, d_C2, SIZE * sizeof(uint64_t), cudaMemcpyDeviceToHost);
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  
  cudaEventElapsedTime(&kernel_time, start, end);
  cudaEventDestroy(start);
  cudaEventDestroy(end);

  check_result(h_cpu_C, h_gpu2_C);
  std::cout << "Kernel 2 time (ms): " << kernel_time << "\n";

  // kernel 3
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  cudaEventRecord(start, 0);

  status = cudaMemcpy(d_A, h_A, SIZE * sizeof(uint64_t), cudaMemcpyHostToDevice);
  status = cudaMemcpy(d_B, h_B, SIZE * sizeof(uint64_t), cudaMemcpyHostToDevice);

  kernel3<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C2);

  cudaMemcpy(h_gpu2_C, d_C2, SIZE * sizeof(uint64_t), cudaMemcpyDeviceToHost);
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  
  cudaEventElapsedTime(&kernel_time, start, end);
  cudaEventDestroy(start);
  cudaEventDestroy(end);

  check_result(h_cpu_C, h_gpu2_C);
  std::cout << "Kernel 3 time (ms): " << kernel_time << "\n";

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C1);
  cudaFree(d_C2);

  free(h_A);
  free(h_B);
  free(h_cpu_C);
  free(h_gpu1_C);
  free(h_gpu2_C);

  return EXIT_SUCCESS;
}
