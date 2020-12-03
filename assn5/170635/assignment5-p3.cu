// Compile: nvcc -g -G -arch=sm_61 -std=c++11 assignment5-p3.cu -o assignment5-p3
// Execute: ./assignment5-p3

#include <cmath>
#include <iostream>
#include <sys/time.h>
#include <cuda.h>

#define SIZE 4096
#define THRESHOLD (0.000001)
#define BLOCK_SIZE 16

using std::cerr;
using std::cout;
using std::endl;

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

__host__ void ATAonCPU(double* M, double* P) {
  for (int k = 0; k < SIZE; k++) {
    for (int i = 0; i < SIZE; i++) {
      for (int j = 0; j < SIZE; j++)
        P[i*SIZE + j] += M[k*SIZE + i] * M[k*SIZE + j];
    }
  }
}

__host__ void check_result(double* Test, double* Ref) {
  double maxdiff = 0, rel_diff = 0;
  int numdiffs = 0;

  for (int i = 0; i < SIZE; i++) {
    for (int j = 0; j < SIZE; j++) {
      rel_diff = (Test[i*SIZE + j] - Ref[i*SIZE + j]);
      if (fabs(rel_diff) > THRESHOLD) {
        numdiffs++;
        if (rel_diff > maxdiff)
          maxdiff = rel_diff;
      }
    }
  }
  if (numdiffs > 0)
    cout << numdiffs << " Diffs found over THRESHOLD " << THRESHOLD << " Max Diff = " << maxdiff
         << "\n";
  else
    cout << "No differences found between base and test versions\n";
}

__host__ void reset(double* h_dev_out){
  for (int i = 0; i < SIZE; i++) {
    for (int j = 0; j < SIZE; j++) {
      h_dev_out[i*SIZE + j] = 0;
    }
  }
}

__global__ void ATAkernel1(double* A, double* B) {
  // TODO: Fill in
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  double val = 0;
  for(int k = 0; k < SIZE; k++){
    val += A[k*SIZE + i] * A[k*SIZE + j];
  }
  B[i*SIZE + j] = val;
}

__global__ void ATAkernel2(double* A, double* B) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  int top_left_i = blockIdx.y * BLOCK_SIZE;
  int top_left_j = blockIdx.x * BLOCK_SIZE;

  double val = 0;

  for(int block_num = 0; block_num < SIZE/BLOCK_SIZE; block_num++){
    __shared__ double mat1[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ double mat2[BLOCK_SIZE][BLOCK_SIZE];

    mat1[threadIdx.y][threadIdx.x] = A[(block_num * BLOCK_SIZE + threadIdx.y)*SIZE + (top_left_i + threadIdx.x)];
    mat2[threadIdx.y][threadIdx.x] = A[(block_num * BLOCK_SIZE + threadIdx.y)*SIZE + (top_left_j + threadIdx.x)];

    __syncthreads();

    for(int k = 0; k < BLOCK_SIZE; k++){
      val += mat1[k][threadIdx.y] * mat2[k][threadIdx.x];
    }

    __syncthreads();
    
  }

  B[i*SIZE + j] = val;
}

__global__ void ATAkernel3(double* A, double* B) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  int top_left_i = blockIdx.y * BLOCK_SIZE;
  int top_left_j = blockIdx.x * BLOCK_SIZE;

  double val = 0;

  for(int block_num = 0; block_num < SIZE/BLOCK_SIZE; block_num++){
    __shared__ double mat1[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ double mat2[BLOCK_SIZE][BLOCK_SIZE];

    mat1[threadIdx.y][threadIdx.x] = A[(block_num * BLOCK_SIZE + threadIdx.y)*SIZE + (top_left_i + threadIdx.x)];
    mat2[threadIdx.y][threadIdx.x] = A[(block_num * BLOCK_SIZE + threadIdx.y)*SIZE + (top_left_j + threadIdx.x)];

    __syncthreads();

    for(int k = 0; k < BLOCK_SIZE; k += 4){
      val += mat1[k][threadIdx.y] * mat2[k][threadIdx.x];
      val += mat1[k+1][threadIdx.y] * mat2[k+1][threadIdx.x];
      val += mat1[k+2][threadIdx.y] * mat2[k+2][threadIdx.x];
      val += mat1[k+3][threadIdx.y] * mat2[k+3][threadIdx.x];
    }

    __syncthreads();
    
  }

  B[i*SIZE + j] = val;
}

int main() {

  cout << "Matrix Size = " << SIZE << "\n";

  double* h_in = new double[SIZE*SIZE];
  double* h_cpu_out = new double[SIZE*SIZE];

  double* h_dev_out = new double[SIZE*SIZE];

  for (int i = 0; i < SIZE; i++) {
    for (int j = 0; j < SIZE; j++) {
      h_in[i*SIZE + j] = i * j * 0.25;
      h_cpu_out[i*SIZE + j] = 0;
      h_dev_out[i*SIZE + j] = 0;
    }
  }

  double clkbegin = rtclock();
  ATAonCPU(h_in, h_cpu_out);
  double clkend = rtclock();
  double cpu_time = clkend - clkbegin;
  cout << "A^T.A on CPU: " << ((2.0 * SIZE * SIZE * SIZE) / cpu_time)
       << " GFLOPS; Time = " << cpu_time * 1000 << " msec" << endl;

  cudaError_t status;
  cudaEvent_t start, end;
  float kernel_time;

  double* d_in;
  double* d_out;
  // TODO: Fill in
  // first kernel
  size_t size = SIZE * SIZE * sizeof(double);
  dim3 threadsPerBlock(32,32);
  dim3 numBlocks(SIZE/threadsPerBlock.x, SIZE/threadsPerBlock.y);

  status = cudaMalloc(&d_in, size);
  if (status != cudaSuccess) {
    fprintf(stderr, "cudaMalloc() failed");
    return EXIT_FAILURE;
  }
  status = cudaMalloc(&d_out, size);
  if (status != cudaSuccess) {
    fprintf(stderr, "cudaMalloc() failed");
    return EXIT_FAILURE;
  }


  cudaEventCreate(&start);
  cudaEventCreate(&end);

  cudaEventRecord(start, 0);

  status = cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);
  if (status != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy() failed");
    return EXIT_FAILURE;
  }

  ATAkernel1<<<numBlocks, threadsPerBlock>>>(d_in, d_out);

  status = cudaMemcpy(h_dev_out, d_out, size, cudaMemcpyDeviceToHost);
  if (status != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy() failed");
    return EXIT_FAILURE;
  }

  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);

  cudaEventElapsedTime(&kernel_time, start, end);
  cudaEventDestroy(start);
  cudaEventDestroy(end);

  check_result(h_cpu_out, h_dev_out);
  cout << "A^T.A version1 on GPU: " << ((2.0 * SIZE * SIZE * SIZE) / (kernel_time * 1.0e-03))
       << " GFLOPS; Time = " << kernel_time << " msec" << endl;


  // second kernel
  reset(h_dev_out);
  threadsPerBlock = dim3(BLOCK_SIZE,BLOCK_SIZE);
  numBlocks = dim3(SIZE/threadsPerBlock.x, SIZE/threadsPerBlock.y);

  cudaEventCreate(&start);
  cudaEventCreate(&end);

  cudaEventRecord(start, 0);

  status = cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);
  if (status != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy() failed");
    return EXIT_FAILURE;
  }

  ATAkernel2<<<numBlocks, threadsPerBlock>>>(d_in, d_out);

  status = cudaMemcpy(h_dev_out, d_out, size, cudaMemcpyDeviceToHost);
  if (status != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy() failed");
    return EXIT_FAILURE;
  }

  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);

  cudaEventElapsedTime(&kernel_time, start, end);
  cudaEventDestroy(start);
  cudaEventDestroy(end);

  check_result(h_cpu_out, h_dev_out);
  cout << "A^T.A version2 on GPU: " << ((2.0 * SIZE * SIZE * SIZE) / (kernel_time * 1.0e-03))
       << " GFLOPS; Time = " << kernel_time << " msec" << endl;

  
  // third kernel
  reset(h_dev_out);
  threadsPerBlock = dim3(BLOCK_SIZE,BLOCK_SIZE);
  numBlocks = dim3(SIZE/threadsPerBlock.x, SIZE/threadsPerBlock.y);

  cudaEventCreate(&start);
  cudaEventCreate(&end);

  cudaEventRecord(start, 0);

  status = cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);
  if (status != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy() failed");
    return EXIT_FAILURE;
  }

  ATAkernel3<<<numBlocks, threadsPerBlock>>>(d_in, d_out);

  status = cudaMemcpy(h_dev_out, d_out, size, cudaMemcpyDeviceToHost);
  if (status != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy() failed");
    return EXIT_FAILURE;
  }

  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);

  cudaEventElapsedTime(&kernel_time, start, end);
  cudaEventDestroy(start);
  cudaEventDestroy(end);

  check_result(h_cpu_out, h_dev_out);
  cout << "A^T.A version3 on GPU: " << ((2.0 * SIZE * SIZE * SIZE) / (kernel_time * 1.0e-03))
       << " GFLOPS; Time = " << kernel_time << " msec" << endl;

  cudaFree(d_in);
  cudaFree(d_out);

  delete[] h_in;
  delete[] h_cpu_out;
  delete[] h_dev_out;

  return EXIT_SUCCESS;
}
