// Compile: nvcc -g -G -arch=sm_61 -std=c++11 assignment5-p5.cu -o assignment5-p5

#include <cmath>
#include <cstdlib>
#include <cuda.h>
#include <iostream>
#include <sys/time.h>

#define N 512
#define THRESHOLD (0.000001)

using std::cerr;
using std::cout;
using std::endl;

// TODO: Edit the function definition as required
__global__ void kernel1(float* d_in, float* d_out) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  if(i>=1 and i<N-1 and j>=1 and j<N-1 and k>=1 and k<N-1){
    d_out[i*N*N + j*N + k] = 0.8 * (d_in[(i-1)*N*N + j*N + k] + d_in[(i+1)*N*N + j*N + k] 
          + d_in[i*N*N + (j-1)*N + k] + d_in[i*N*N + (j+1)*N + k] + d_in[i*N*N + j*N + (k-1)]
          + d_in[i*N*N + j*N + (k+1)]);
  }
}

// TODO: Edit the function definition as required
__global__ void kernel2(float* d_in, float* d_out) {
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.z * blockDim.z + threadIdx.z;

  if(i>=1 and i<N-1 and j>=1 and j<N-1 and k>=1 and k<N-1){
    d_out[i*N*N + j*N + k] = 0.8 * (d_in[(i-1)*N*N + j*N + k] + d_in[(i+1)*N*N + j*N + k] 
          + d_in[i*N*N + (j-1)*N + k] + d_in[i*N*N + (j+1)*N + k] + d_in[i*N*N + j*N + (k-1)]
          + d_in[i*N*N + j*N + (k+1)]);
  }
}

__global__ void kernel3(float* d_in, float* d_out) {
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.z * blockDim.z + threadIdx.z;

  int tj = threadIdx.y;
  int tk = threadIdx.x;
  int ti = threadIdx.z;

  __shared__ float mat[32][8][4];
  mat[tk][tj][ti] = d_in[i*N*N + j*N + k];

  __syncthreads();

  if(i>=1 and i<N-1 and j>=1 and j<N-1 and k>=1 and k<N-1){
    float val = 0;

    if(ti < 1)
      val += d_in[(i-1)*N*N + j*N + k];
    else
      val += mat[tk][tj][ti-1];

    if(ti + 1 > 3)
      val += d_in[(i+1)*N*N + j*N + k];
    else
      val += mat[tk][tj][ti+1];

    if(tj < 1)
      val += d_in[i*N*N + (j-1)*N + k];
    else
      val += mat[tk][tj-1][ti];

    if(tj + 1 > 7)
      val += d_in[i*N*N + (j+1)*N + k];
    else
      val += mat[tk][tj+1][ti];
    
    if(tk < 1)
      val += d_in[i*N*N + j*N + k-1];
    else
      val += mat[tk-1][tj][ti];

    if(tk + 1 > 31)
      val += d_in[i*N*N + j*N + k+1];
    else
      val += mat[tk+1][tj][ti];

    d_out[i*N*N + j*N + k] = 0.8 * val;
  }

  __syncthreads();
}

// TODO: Edit the function definition as required
__host__ void stencil(float* h_in, float* h_out) {
  for(int i = 1; i < N-1; i++){
    for(int j = 1; j < N-1; j++){
      for(int k = 1; k < N-1; k++){
        h_out[i*N*N + j*N + k] = 0.8 * (h_in[(i-1)*N*N + j*N + k] + h_in[(i+1)*N*N + j*N + k] 
          + h_in[i*N*N + (j-1)*N + k] + h_in[i*N*N + (j+1)*N + k] + h_in[i*N*N + j*N + (k-1)]
          + h_in[i*N*N + j*N + (k+1)]);
      }
    }
  }
}

__host__ void check_result(float* w_ref, float* w_opt) {
  double maxdiff = 0.0, this_diff = 0.0;
  int numdiffs = 0;

  for (uint64_t i = 0; i < N; i++) {
    for (uint64_t j = 0; j < N; j++) {
      for (uint64_t k = 0; k < N; k++) {
        this_diff = w_ref[i + N * j + N * N * k] - w_opt[i + N * j + N * N * k];
        if (std::fabs(this_diff) > THRESHOLD) {
          numdiffs++;
          if (this_diff > maxdiff) {
            maxdiff = this_diff;
          }
        }
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
  uint64_t SIZE = N * N * N;

  float*h_in, *h_cpu_out, *h_gpu1_out, *h_gpu2_out, *h_gpu3_out;

  h_in = (float*)malloc(SIZE * sizeof(float));
  h_cpu_out = (float*)malloc(SIZE * sizeof(float));
  h_gpu1_out = (float*)malloc(SIZE * sizeof(float));
  h_gpu2_out = (float*)malloc(SIZE * sizeof(float));
  h_gpu3_out = (float*)malloc(SIZE * sizeof(float));

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < N; k++) {
        h_in[i * N * N + j * N + k] = rand() % 64;
        h_cpu_out[i * N * N + j * N + k] = 0;
        h_gpu1_out[i * N * N + j * N + k] = 0;
        h_gpu2_out[i * N * N + j * N + k] = 0;
        h_gpu3_out[i * N * N + j * N + k] = 0;
      }
    }
  }

  double clkbegin = rtclock();
  stencil(h_in, h_cpu_out);
  double clkend = rtclock();
  double cpu_time = clkend - clkbegin;
  cout << "Stencil time on CPU: " << cpu_time * 1000 << " msec" << endl;

  cudaError_t status;
  cudaEvent_t start, end;
  // TODO: Fill in kernel1
  // TODO: Adapt check_result() and invoke
  float *d_in, *d_out1;
  dim3 threadsPerBlock(32,32,1);
  dim3 numBlocks(N/threadsPerBlock.x, N/threadsPerBlock.y, N/threadsPerBlock.z);

  status = cudaMalloc(&d_in, SIZE * sizeof(float));
  if (status != cudaSuccess) {
    fprintf(stderr, "cudaMalloc() failed");
    return EXIT_FAILURE;
  }
  status = cudaMalloc(&d_out1, SIZE * sizeof(float));
  if (status != cudaSuccess) {
    fprintf(stderr, "cudaMalloc() failed");
    return EXIT_FAILURE;
  }

  cudaEventCreate(&start);
  cudaEventCreate(&end);

  cudaEventRecord(start, 0);

  status = cudaMemcpy(d_in, h_in, SIZE * sizeof(float), cudaMemcpyHostToDevice);
  if (status != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy() failed");
    return EXIT_FAILURE;
  }
  status = cudaMemcpy(d_out1, h_gpu1_out, SIZE * sizeof(float), cudaMemcpyHostToDevice);
  if (status != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy() failed");
    return EXIT_FAILURE;
  }

  kernel1<<<numBlocks, threadsPerBlock>>>(d_in, d_out1);

  status = cudaMemcpy(h_gpu1_out, d_out1, SIZE * sizeof(float), cudaMemcpyDeviceToHost);
  if (status != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy() failed");
    return EXIT_FAILURE;
  }

  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);

  float kernel_time;
  cudaEventElapsedTime(&kernel_time, start, end);
  cudaEventDestroy(start);
  cudaEventDestroy(end);

  check_result(h_cpu_out, h_gpu1_out);
  std::cout << "Kernel 1 time (ms): " << kernel_time << "\n";

  // TODO: Fill in kernel2
  // TODO: Adapt check_result() and invoke
  float *d_out2;

  status = cudaMalloc(&d_out2, SIZE * sizeof(float));
  if (status != cudaSuccess) {
    fprintf(stderr, "cudaMalloc() failed");
    return EXIT_FAILURE;
  }

  cudaEventCreate(&start);
  cudaEventCreate(&end);

  cudaEventRecord(start, 0);

  status = cudaMemcpy(d_in, h_in, SIZE * sizeof(float), cudaMemcpyHostToDevice);
  if (status != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy() failed");
    return EXIT_FAILURE;
  }
  status = cudaMemcpy(d_out2, h_gpu2_out, SIZE * sizeof(float), cudaMemcpyHostToDevice);
  if (status != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy() failed");
    return EXIT_FAILURE;
  }

  kernel2<<<numBlocks, threadsPerBlock>>>(d_in, d_out2);

  status = cudaMemcpy(h_gpu2_out, d_out2, SIZE * sizeof(float), cudaMemcpyDeviceToHost);
  if (status != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy() failed");
    return EXIT_FAILURE;
  }

  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);

  cudaEventElapsedTime(&kernel_time, start, end);
  cudaEventDestroy(start);
  cudaEventDestroy(end);

  check_result(h_cpu_out, h_gpu2_out);
  std::cout << "Kernel 2 time (ms): " << kernel_time << "\n";

  // kernel 3
  float *d_out3;
  threadsPerBlock = dim3(32,8,4);
  numBlocks = dim3(N/threadsPerBlock.x, N/threadsPerBlock.y, N/threadsPerBlock.z);

  status = cudaMalloc(&d_out3, SIZE * sizeof(float));
  if (status != cudaSuccess) {
    fprintf(stderr, "cudaMalloc() failed");
    return EXIT_FAILURE;
  }

  cudaEventCreate(&start);
  cudaEventCreate(&end);

  cudaEventRecord(start, 0);

  status = cudaMemcpy(d_in, h_in, SIZE * sizeof(float), cudaMemcpyHostToDevice);
  if (status != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy() failed");
    return EXIT_FAILURE;
  }
  status = cudaMemcpy(d_out3, h_gpu3_out, SIZE * sizeof(float), cudaMemcpyHostToDevice);
  if (status != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy() failed");
    return EXIT_FAILURE;
  }

  kernel3<<<numBlocks, threadsPerBlock>>>(d_in, d_out3);

  status = cudaMemcpy(h_gpu3_out, d_out3, SIZE * sizeof(float), cudaMemcpyDeviceToHost);
  if (status != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy() failed");
    return EXIT_FAILURE;
  }

  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);

  cudaEventElapsedTime(&kernel_time, start, end);
  cudaEventDestroy(start);
  cudaEventDestroy(end);

  check_result(h_cpu_out, h_gpu3_out);
  std::cout << "Kernel 3 time (ms): " << kernel_time << "\n";

  // TODO: Free memory
  cudaFree(d_in);
  cudaFree(d_out1);
  cudaFree(d_out2);
  cudaFree(d_out3);

  free(h_in);
  free(h_cpu_out);
  free(h_gpu1_out);
  free(h_gpu2_out);
  free(h_gpu3_out);

  return EXIT_SUCCESS;
}
