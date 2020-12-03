// Compile: nvcc -g -G -arch=sm_61 -std=c++11 assignment5-p2.cu -o assignment5-p2
// Execute: ./assignment5-p2

#include <algorithm>
#include <cuda.h>
#include <iostream>
#include <sys/time.h>

#define THRESHOLD (0.000001)

#define N (1 << 24)
#define CHUNK_SIZE 2048
#define CHUNK_SIZE2 2048

using std::cerr;
using std::cout;
using std::endl;

__host__ void host_excl_prefix_sum(float* h_A, float* h_O) {
  h_O[0] = 0;
  for (int i = 1; i < N; i++) {
    h_O[i] = h_O[i - 1] + h_A[i - 1];
  }
}

__global__ void kernel_excl_prefix_sum_ver1_1(float* d_in, float* d_out) {
  // TODO: Fill in
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  i *= CHUNK_SIZE;

  if(i < N){
    for(int j = i+1; j < i+CHUNK_SIZE; j++){
      d_out[j] = d_out[j-1] + d_in[j-1];
    }
  }
}

__global__ void kernel_excl_prefix_sum_ver1_2(float* d_in, float* d_out, long long int curr_chunk) {
  
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int times = curr_chunk/CHUNK_SIZE;
  int chunk_num = i/times;
  int chunk_part = i%times;
  int c = (2*chunk_num + 1)*curr_chunk;

  float sum = d_out[c-1] + d_in[c-1];
  i = c + (curr_chunk * chunk_part)/times;

  int upper_limit = i + CHUNK_SIZE;
  if(i < N){
    for(int j = i; j < upper_limit; j++){
      d_out[j] += sum;
    }
  }
}

__global__ void kernel_excl_prefix_sum_ver2_1(float* d_in, float* d_out) {
  // TODO: Fill in
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  i *= CHUNK_SIZE2;

  if(i < N){
    for(int j = i+1; j < i+CHUNK_SIZE2; j++){
      d_out[j] = d_out[j-1] + d_in[j-1];
    }
  }
}

__global__ void kernel_excl_prefix_sum_ver2_2(float* d_in, float* d_out, long long int curr_chunk) {
  
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int times = curr_chunk/CHUNK_SIZE2;
  int chunk_num = i/times;
  int chunk_part = i%times;
  int c = (2*chunk_num + 1)*curr_chunk;

  float sum = d_out[c-1] + d_in[c-1];
  i = c + (curr_chunk * chunk_part)/times;

  int upper_limit = i + CHUNK_SIZE2;
  if(i < N){
    for(int j = i; j + 3 < upper_limit; j += 4){
      d_out[j] += sum;
      d_out[j+1] += sum;
      d_out[j+2] += sum;
      d_out[j+3] += sum;
    }
  }
}

__host__ void check_result(float* w_ref, float* w_opt) {
  double maxdiff = 0.0, this_diff = 0.0;
  int numdiffs = 0;

  for (int i = 0; i < N; i++) {
    this_diff = w_ref[i] - w_opt[i];
    if (fabs(this_diff) > THRESHOLD) {
      numdiffs++;
      if (this_diff > maxdiff)
        maxdiff = this_diff;
    }
  }

  if (numdiffs > 0) {
    cout << numdiffs << " Diffs found over threshold " << THRESHOLD << "; Max Diff = " << maxdiff
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
  size_t size = N * sizeof(float);

  float* h_in = (float*)malloc(size);
  std::fill_n(h_in, N, 1);

  float* h_excl_sum_out = (float*)malloc(size);
  std::fill_n(h_excl_sum_out, N, 0);

  double clkbegin = rtclock();
  host_excl_prefix_sum(h_in, h_excl_sum_out);
  double clkend = rtclock();
  double time = clkend - clkbegin; // seconds
  cout << "Serial time on CPU: " << time * 1000 << " msec" << endl;

  float* h_dev_result = (float*)malloc(size);
  std::fill_n(h_dev_result, N, 0);
  float* d_k1_in;
  float* d_k1_out;
  cudaError_t status;
  cudaEvent_t start, end;
  // TODO: Fill in

  status = cudaMalloc(&d_k1_in, size);
  if (status != cudaSuccess) {
    fprintf(stderr, "cudaMalloc() failed");
    return EXIT_FAILURE;
  }
  status = cudaMalloc(&d_k1_out, size);
  if (status != cudaSuccess) {
    fprintf(stderr, "cudaMalloc() failed");
    return EXIT_FAILURE;
  }


  cudaEventCreate(&start);
  cudaEventCreate(&end);

  cudaEventRecord(start, 0);

  status = cudaMemcpy(d_k1_in, h_in, size, cudaMemcpyHostToDevice);
  if (status != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy() failed");
    return EXIT_FAILURE;
  }
  status = cudaMemcpy(d_k1_out, h_dev_result, size, cudaMemcpyHostToDevice);
  if (status != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy() failed");
    return EXIT_FAILURE;
  }

  dim3 threadsPerBlock(256);
  dim3 numBlocks(((N/CHUNK_SIZE) + threadsPerBlock.x - 1)/threadsPerBlock.x);
  kernel_excl_prefix_sum_ver1_1<<<numBlocks, threadsPerBlock>>>(d_k1_in, d_k1_out);

  numBlocks = dim3((N/(2*CHUNK_SIZE) + threadsPerBlock.x - 1)/threadsPerBlock.x);
  long long int curr_chunk = CHUNK_SIZE;
  while(curr_chunk != N){
    kernel_excl_prefix_sum_ver1_2<<<numBlocks, threadsPerBlock>>>(d_k1_in, d_k1_out, curr_chunk);
    curr_chunk *= 2;
  }


  status = cudaMemcpy(h_dev_result, d_k1_out, size, cudaMemcpyDeviceToHost);
  if (status != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy() failed");
    return EXIT_FAILURE;
  }

  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);

  float k_time; // ms
  cudaEventElapsedTime(&k_time, start, end);
  cudaEventDestroy(start);
  cudaEventDestroy(end);

  check_result(h_excl_sum_out, h_dev_result);
  cout << "Kernel1 time on GPU: " << k_time << " msec" << endl;

  // kernel 2
  std::fill_n(h_dev_result, N, 0);
  float* d_k2_in;
  float* d_k2_out;

  status = cudaMalloc(&d_k2_in, size);
  if (status != cudaSuccess) {
    fprintf(stderr, "cudaMalloc() failed");
    return EXIT_FAILURE;
  }
  status = cudaMalloc(&d_k2_out, size);
  if (status != cudaSuccess) {
    fprintf(stderr, "cudaMalloc() failed");
    return EXIT_FAILURE;
  }

  cudaEventCreate(&start);
  cudaEventCreate(&end);

  cudaEventRecord(start, 0);

  status = cudaMemcpy(d_k2_in, h_in, size, cudaMemcpyHostToDevice);
  if (status != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy() failed");
    return EXIT_FAILURE;
  }
  status = cudaMemcpy(d_k2_out, h_dev_result, size, cudaMemcpyHostToDevice);
  if (status != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy() failed");
    return EXIT_FAILURE;
  }

  threadsPerBlock = dim3(256);
  numBlocks = dim3(((N/CHUNK_SIZE2) + threadsPerBlock.x - 1)/threadsPerBlock.x);
  kernel_excl_prefix_sum_ver2_1<<<numBlocks, threadsPerBlock>>>(d_k2_in, d_k2_out);

  threadsPerBlock = dim3(256);
  numBlocks = dim3((N/(2*CHUNK_SIZE2) + threadsPerBlock.x - 1)/threadsPerBlock.x);
  curr_chunk = CHUNK_SIZE2;
  while(curr_chunk != N){
    kernel_excl_prefix_sum_ver2_2<<<numBlocks, threadsPerBlock>>>(d_k2_in, d_k2_out, curr_chunk);
    curr_chunk *= 2;
  }

  status = cudaMemcpy(h_dev_result, d_k2_out, size, cudaMemcpyDeviceToHost);
  if (status != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy() failed");
    return EXIT_FAILURE;
  }

  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);

  cudaEventElapsedTime(&k_time, start, end);
  cudaEventDestroy(start);
  cudaEventDestroy(end);

  check_result(h_excl_sum_out, h_dev_result);
  cout << "Kernel2 time on GPU: " << k_time << " msec" << endl;

  // Free device memory
  cudaFree(d_k1_in);
  cudaFree(d_k1_out);
  cudaFree(d_k2_in);
  cudaFree(d_k2_out);

  free(h_in);
  free(h_excl_sum_out);
  free(h_dev_result);

  return EXIT_SUCCESS;
}
