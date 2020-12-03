// Compile: nvcc -g -G -arch=sm_61 -std=c++11 assignment5-p1.cu -o assignment5-p1
// Execute: ./assignment5-p1

#include <cmath>
#include <cstdint>
#include <cuda.h>
#include <iostream>
#include <new>
#include <sys/time.h>

#define THRESHOLD (0.000001)

#define SIZE1 8192
#define SIZE2 8200
#define ITER 100

using std::cerr;
using std::cout;
using std::endl;

__global__ void kernel1(double* d_k1) {
  // TODO: Fill in
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if(j < SIZE1 - 1){
    for(int k = 0; k < ITER; k++){
      for(int i = 1; i < SIZE1 - 1; i++){
        d_k1[i*SIZE1 + (j+1)] = d_k1[(i-1)*SIZE1 + (j+1)] + d_k1[i*SIZE1 + (j+1)] + 
          d_k1[(i+1)*SIZE1 + (j+1)];
      }
    }
  }
}

__global__ void kernel2(double* d_k2) {
  // TODO: Fill in
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if(j < SIZE2 - 1){
    for(int k = 0; k < ITER; k++){
      for(int i = 1; i < SIZE2 - 1; i++){
        d_k2[i*SIZE2 + (j+1)] = d_k2[(i-1)*SIZE2 + (j+1)] + d_k2[i*SIZE2 + (j+1)] + 
          d_k2[(i+1)*SIZE2 + (j+1)];
      }
    }
  }
}

// 2-way unrolling
__global__ void kernel3(double* d_k3) {
  // TODO: Fill in
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if(j < SIZE2 - 1){
    for(int k = 0; k < ITER; k++){
      for(int i = 1; i < SIZE2 - 1; i += 2){
        double w, x, y, z;
        w = d_k3[(i-1)*SIZE2 + (j+1)];
        x = d_k3[(i)*SIZE2 + (j+1)];
        y = d_k3[(i+1)*SIZE2 + (j+1)];
        z = d_k3[(i+2)*SIZE2 + (j+1)];

        x = w + x + y;
        y = x + y + z;

        d_k3[(i)*SIZE2 + (j+1)] = x;
        d_k3[(i+1)*SIZE2 + (j+1)] = y;
      }
    }
  }
}

__host__ void serial(double** h_ser) {
  for (int k = 0; k < ITER; k++) {
    for (int i = 1; i < (SIZE1 - 1); i++) {
      for (int j = 0; j < (SIZE1 - 1); j++) {
        h_ser[i][j + 1] = (h_ser[i - 1][j + 1] + h_ser[i][j + 1] + h_ser[i + 1][j + 1]);
      }
    }
  }
}

__host__ void check_result(double** w_ref, double** w_opt, uint64_t size) {
  double maxdiff = 0.0, this_diff = 0.0;
  int numdiffs = 0;

  for (uint64_t i = 0; i < size; i++) {
    for (uint64_t j = 0; j < size; j++) {
      this_diff = w_ref[i][j] - w_opt[i][j];
      if (fabs(this_diff) > THRESHOLD) {
        numdiffs++;
        if (this_diff > maxdiff)
          maxdiff = this_diff;
      }
    }
  }

  if (numdiffs > 0) {
    cout << numdiffs << " Diffs found over THRESHOLD " << THRESHOLD << "; Max Diff = " << maxdiff
         << endl;
  } else {
    cout << "No differences found between base and test versions\n";
  }
}

__host__ double rtclock() { // Seconds
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
  double** h_ser = new double*[SIZE1];

  double** h_k1 = new double*[SIZE1];
  double** h_k2 = new double*[SIZE2];
  double** h_k3 = new double*[SIZE2];

  for(int i = 0; i < SIZE1; i++){
    h_ser[i] = new double[SIZE1];
  }

  h_k1[0] = new double[SIZE1 * SIZE1];
  h_k2[0] = new double[SIZE2 * SIZE2];
  h_k3[0] = new double[SIZE2 * SIZE2];
  for (int i = 1; i < SIZE1; i++) {
    h_k1[i] = h_k1[i-1] + SIZE1;
  }

  for (int i = 0; i < SIZE1; i++) {
    for (int j = 0; j < SIZE1; j++) {
      h_ser[i][j] = 1;
      h_k1[i][j] = 1;
    }
  }

  for (int i = 1; i < SIZE2; i++) {
    h_k2[i] = h_k2[i-1] + SIZE2;
    h_k3[i] = h_k3[i-1] + SIZE2;
  }

  for (int i = 0; i < SIZE2; i++) {
    for (int j = 0; j < SIZE2; j++) {
      h_k2[i][j] = 1;
      h_k3[i][j] = 1;
    }
  }

  double clkbegin = rtclock();
  serial(h_ser);
  double clkend = rtclock();
  double time = clkend - clkbegin; // seconds
  cout << "Serial code on CPU: " << ((2.0 * SIZE1 * SIZE1 * ITER) / time)
       << " GFLOPS; Time = " << time * 1000 << " msec" << endl;

  cudaError_t status;
  cudaEvent_t start, end;
  float k1_time, k2_time, k3_time; // ms

  double* d_k1;
  // TODO: Fill in
  size_t size = SIZE1 * SIZE1 * sizeof(double);
  dim3 threadsPerBlock(1024);
  dim3 numBlocks((SIZE1+threadsPerBlock.x-1)/threadsPerBlock.x);

  status = cudaMalloc(&d_k1, size);
  if (status != cudaSuccess) {
    fprintf(stderr, "cudaMalloc() failed");
    return EXIT_FAILURE;
  }

  cudaEventCreate(&start);
  cudaEventCreate(&end);

  cudaEventRecord(start, 0);

  status = cudaMemcpy(d_k1, h_k1[0], size, cudaMemcpyHostToDevice);
  if (status != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy() failed");
    return EXIT_FAILURE;
  }

  kernel1<<<numBlocks, threadsPerBlock>>>(d_k1);

  status = cudaMemcpy(h_k1[0], d_k1, size, cudaMemcpyDeviceToHost);
  if (status != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy() failed");
    return EXIT_FAILURE;
  }

  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);

  cudaEventElapsedTime(&k1_time, start, end);
  cudaEventDestroy(start);
  cudaEventDestroy(end);


  check_result(h_ser, h_k1, SIZE1);
  cout << "Kernel 1 on GPU: " << ((2.0 * SIZE1 * SIZE1 * ITER) / (k1_time * 1.0e-3))
       << " GFLOPS; Time = " << k1_time << " msec" << endl;

  double* d_k2;
  // TODO: Fill in
  size = SIZE2 * SIZE2 * sizeof(double);
  threadsPerBlock = dim3(32);
  numBlocks = dim3((SIZE2+threadsPerBlock.x-1)/threadsPerBlock.x);

  status = cudaMalloc(&d_k2, size);
  if (status != cudaSuccess) {
    fprintf(stderr, "cudaMalloc() failed");
    return EXIT_FAILURE;
  }

  cudaEventCreate(&start);
  cudaEventCreate(&end);

  cudaEventRecord(start, 0);

  status = cudaMemcpy(d_k2, h_k2[0], size, cudaMemcpyHostToDevice);
  if (status != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy() failed");
    return EXIT_FAILURE;
  }

  kernel2<<<numBlocks, threadsPerBlock>>>(d_k2);

  status = cudaMemcpy(h_k2[0], d_k2, size, cudaMemcpyDeviceToHost);
  if (status != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy() failed");
    return EXIT_FAILURE;
  }

  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);

  cudaEventElapsedTime(&k2_time, start, end);
  cudaEventDestroy(start);
  cudaEventDestroy(end);


  cout << "Kernel 2 on GPU: " << ((2.0 * SIZE2 * SIZE2 * ITER) / (k2_time * 1.0e-3))
       << " GFLOPS; Time = " << k2_time << " msec" << endl;

  // kernel 3
  double* d_k3;
  // TODO: Fill in
  size = SIZE2 * SIZE2 * sizeof(double);
  threadsPerBlock = dim3(32);
  numBlocks = dim3((SIZE2+threadsPerBlock.x-1)/threadsPerBlock.x);

  status = cudaMalloc(&d_k3, size);
  if (status != cudaSuccess) {
    fprintf(stderr, "cudaMalloc() failed");
    return EXIT_FAILURE;
  }

  cudaEventCreate(&start);
  cudaEventCreate(&end);

  cudaEventRecord(start, 0);

  status = cudaMemcpy(d_k3, h_k3[0], size, cudaMemcpyHostToDevice);
  if (status != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy() failed");
    return EXIT_FAILURE;
  }

  kernel3<<<numBlocks, threadsPerBlock>>>(d_k3);

  status = cudaMemcpy(h_k3[0], d_k3, size, cudaMemcpyDeviceToHost);
  if (status != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy() failed");
    return EXIT_FAILURE;
  }

  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);

  cudaEventElapsedTime(&k3_time, start, end);
  cudaEventDestroy(start);
  cudaEventDestroy(end);

  // checking whether kernel2 and kernel3 produce same result
  check_result(h_k2, h_k3, SIZE2);
  cout << "Kernel 3 on GPU: " << ((2.0 * SIZE2 * SIZE2 * ITER) / (k3_time * 1.0e-3))
       << " GFLOPS; Time = " << k3_time << " msec" << endl;

  cudaFree(d_k1);
  cudaFree(d_k2);
  cudaFree(d_k3);

  for(int i = 0; i < SIZE1; i++){
    delete[] h_ser[i];
  }

  delete[] h_k1[0];

  delete[] h_ser;
  delete[] h_k1;

  delete[] h_k2[0];
  delete[] h_k3[0];
  
  delete[] h_k2;
  delete[] h_k3;

  return EXIT_SUCCESS;
}
