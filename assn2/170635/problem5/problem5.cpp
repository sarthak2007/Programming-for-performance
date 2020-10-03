/**
 * g++ -o problem5 problem5.cpp -pthread
 * ./problem5 <block_size> <num_threads>
 */

// TODO: This file is just a template, feel free to modify it to suit your needs

#include <cstring>
#include <iostream>
#include <pthread.h>
#include <sys/time.h>

using std::cout;
using std::endl;

const uint16_t MAT_SIZE = 4096;

void sequential_matmul();
void parallel_matmul();
// TODO: Other function definitions
void sequential_matmul_opt_1();
void sequential_matmul_opt_2();
void sequential_matmul_opt_3();
void sequential_matmul_opt_4();
void sequential_matmul_opt_5();
void sequential_matmul_opt_6();
void parallel_matmul_opt();

double rtclock();
void check_result(uint64_t*, uint64_t*);
const double THRESHOLD = 0.0000001;

uint64_t* matrix_A;
uint64_t* matrix_B;
uint64_t* sequential_C;
uint64_t* parallel_C;
uint64_t* sequential_opt_C_1;
uint64_t* sequential_opt_C_2;
uint64_t* sequential_opt_C_3;
uint64_t* sequential_opt_C_4;
uint64_t* sequential_opt_C_5;
uint64_t* sequential_opt_C_6;
uint64_t* parallel_opt_C;


uint16_t block_size; // taken from command line arg
uint16_t NUM_THREADS; // taken from command line arg
uint16_t unrolling_factor1 = 2;
uint16_t unrolling_factor2 = 4;
uint16_t unrolling_factor3 = 8;
uint16_t unrolling_factor4 = 16;
uint16_t unrolling_factor5 = 32;


void sequential_matmul() {
	int i, j, k;
	for (i = 0; i < MAT_SIZE; i++){
		for (j = 0; j < MAT_SIZE; j++){
			uint64_t temp = 0;
			for (k = 0; k < MAT_SIZE; k++)
				temp += (matrix_A[i * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j]);
			sequential_C[i * MAT_SIZE + j] = temp;
		}
	}
}

void* parallel_matmul(void* thread_id){
	int id = (intptr_t)thread_id;
	int i, j, k;
	for (i = id*MAT_SIZE/NUM_THREADS; i < (id+1)*MAT_SIZE/NUM_THREADS; i++){
		for (j = 0; j < MAT_SIZE; j++){
			uint64_t temp = 0;
			for (k = 0; k < MAT_SIZE; k++)
				temp += (matrix_A[i * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j]);
			parallel_C[i * MAT_SIZE + j] = temp;
		}
	}
    pthread_exit(NULL);
}

void sequential_matmul_opt_1() {
  // TODO:
	int i, j, k;
	for (i = 0; i < MAT_SIZE; i += block_size){
		for (j = 0; j < MAT_SIZE; j += block_size){
			for (k = 0; k < MAT_SIZE; k += block_size)
				for(int i1 = i; i1 < i + block_size; i1++)
					for(int j1 = j; j1 < j + block_size; j1++){
						uint64_t temp = 0;
						for(int k1 = k; k1 + unrolling_factor1 - 1 < k + block_size; k1 += unrolling_factor1){
							temp += (matrix_A[i1 * MAT_SIZE + k1] * matrix_B[k1 * MAT_SIZE + j1]);
							temp += (matrix_A[i1 * MAT_SIZE + k1+1] * matrix_B[(k1+1) * MAT_SIZE + j1]);
						}
						for(int k1 = k + unrolling_factor1 * (block_size/unrolling_factor1); k1 < k + block_size; k1++){
							temp += (matrix_A[i1 * MAT_SIZE + k1] * matrix_B[k1 * MAT_SIZE + j1]);
						}
						sequential_opt_C_1[i1 * MAT_SIZE + j1] += temp;
					}
		}
	}
}

void sequential_matmul_opt_2() {
  // TODO:
	int i, j, k;
	for (i = 0; i < MAT_SIZE; i += block_size){
		for (j = 0; j < MAT_SIZE; j += block_size){
			for (k = 0; k < MAT_SIZE; k += block_size)
				for(int i1 = i; i1 < i + block_size; i1++)
					for(int j1 = j; j1 < j + block_size; j1++){
						uint64_t temp = 0;
						for(int k1 = k; k1 + unrolling_factor2 - 1 < k + block_size; k1 += unrolling_factor2){
							temp += (matrix_A[i1 * MAT_SIZE + k1] * matrix_B[k1 * MAT_SIZE + j1]);
							temp += (matrix_A[i1 * MAT_SIZE + k1+1] * matrix_B[(k1+1) * MAT_SIZE + j1]);
							temp += (matrix_A[i1 * MAT_SIZE + k1+2] * matrix_B[(k1+2) * MAT_SIZE + j1]);
							temp += (matrix_A[i1 * MAT_SIZE + k1+3] * matrix_B[(k1+3) * MAT_SIZE + j1]);
						}
						for(int k1 = k + unrolling_factor2 * (block_size/unrolling_factor2); k1 < k + block_size; k1++){
							temp += (matrix_A[i1 * MAT_SIZE + k1] * matrix_B[k1 * MAT_SIZE + j1]);
						}
						sequential_opt_C_2[i1 * MAT_SIZE + j1] += temp;
					}
		}
	}
}

void sequential_matmul_opt_3() {
  // TODO:
	int i, j, k;
	for (i = 0; i < MAT_SIZE; i += block_size){
		for (j = 0; j < MAT_SIZE; j += block_size){
			for (k = 0; k < MAT_SIZE; k += block_size)
				for(int i1 = i; i1 < i + block_size; i1++)
					for(int j1 = j; j1 < j + block_size; j1++){
						uint64_t temp = 0;
						for(int k1 = k; k1 + unrolling_factor3 - 1 < k + block_size; k1 += unrolling_factor3){
							temp += (matrix_A[i1 * MAT_SIZE + k1] * matrix_B[k1 * MAT_SIZE + j1]);
							temp += (matrix_A[i1 * MAT_SIZE + k1+1] * matrix_B[(k1+1) * MAT_SIZE + j1]);
							temp += (matrix_A[i1 * MAT_SIZE + k1+2] * matrix_B[(k1+2) * MAT_SIZE + j1]);
							temp += (matrix_A[i1 * MAT_SIZE + k1+3] * matrix_B[(k1+3) * MAT_SIZE + j1]);
							temp += (matrix_A[i1 * MAT_SIZE + k1+4] * matrix_B[(k1+4) * MAT_SIZE + j1]);
							temp += (matrix_A[i1 * MAT_SIZE + k1+5] * matrix_B[(k1+5) * MAT_SIZE + j1]);
							temp += (matrix_A[i1 * MAT_SIZE + k1+6] * matrix_B[(k1+6) * MAT_SIZE + j1]);
							temp += (matrix_A[i1 * MAT_SIZE + k1+7] * matrix_B[(k1+7) * MAT_SIZE + j1]);
						}
						for(int k1 = k + unrolling_factor3 * (block_size/unrolling_factor3); k1 < k + block_size; k1++){
							temp += (matrix_A[i1 * MAT_SIZE + k1] * matrix_B[k1 * MAT_SIZE + j1]);
						}
						sequential_opt_C_3[i1 * MAT_SIZE + j1] += temp;
					}
		}
	}
}

void sequential_matmul_opt_4() {
  // TODO:
	int i, j, k;
	for (i = 0; i < MAT_SIZE; i += block_size){
		for (j = 0; j < MAT_SIZE; j += block_size){
			for (k = 0; k < MAT_SIZE; k += block_size)
				for(int i1 = i; i1 < i + block_size; i1++)
					for(int j1 = j; j1 < j + block_size; j1++){
						uint64_t temp = 0;
						for(int k1 = k; k1 + unrolling_factor4 - 1 < k + block_size; k1 += unrolling_factor4){
							temp += (matrix_A[i1 * MAT_SIZE + k1] * matrix_B[k1 * MAT_SIZE + j1]);
							temp += (matrix_A[i1 * MAT_SIZE + k1+1] * matrix_B[(k1+1) * MAT_SIZE + j1]);
							temp += (matrix_A[i1 * MAT_SIZE + k1+2] * matrix_B[(k1+2) * MAT_SIZE + j1]);
							temp += (matrix_A[i1 * MAT_SIZE + k1+3] * matrix_B[(k1+3) * MAT_SIZE + j1]);
							temp += (matrix_A[i1 * MAT_SIZE + k1+4] * matrix_B[(k1+4) * MAT_SIZE + j1]);
							temp += (matrix_A[i1 * MAT_SIZE + k1+5] * matrix_B[(k1+5) * MAT_SIZE + j1]);
							temp += (matrix_A[i1 * MAT_SIZE + k1+6] * matrix_B[(k1+6) * MAT_SIZE + j1]);
							temp += (matrix_A[i1 * MAT_SIZE + k1+7] * matrix_B[(k1+7) * MAT_SIZE + j1]);
							temp += (matrix_A[i1 * MAT_SIZE + k1+8] * matrix_B[(k1+8) * MAT_SIZE + j1]);
							temp += (matrix_A[i1 * MAT_SIZE + k1+9] * matrix_B[(k1+9) * MAT_SIZE + j1]);
							temp += (matrix_A[i1 * MAT_SIZE + k1+10] * matrix_B[(k1+10) * MAT_SIZE + j1]);
							temp += (matrix_A[i1 * MAT_SIZE + k1+11] * matrix_B[(k1+11) * MAT_SIZE + j1]);
							temp += (matrix_A[i1 * MAT_SIZE + k1+12] * matrix_B[(k1+12) * MAT_SIZE + j1]);
							temp += (matrix_A[i1 * MAT_SIZE + k1+13] * matrix_B[(k1+13) * MAT_SIZE + j1]);
							temp += (matrix_A[i1 * MAT_SIZE + k1+14] * matrix_B[(k1+14) * MAT_SIZE + j1]);
							temp += (matrix_A[i1 * MAT_SIZE + k1+15] * matrix_B[(k1+15) * MAT_SIZE + j1]);
						}
						for(int k1 = k + unrolling_factor4 * (block_size/unrolling_factor4); k1 < k + block_size; k1++){
							temp += (matrix_A[i1 * MAT_SIZE + k1] * matrix_B[k1 * MAT_SIZE + j1]);
						}
						sequential_opt_C_4[i1 * MAT_SIZE + j1] += temp;
					}
		}
	}
}

void sequential_matmul_opt_5() {
  // TODO:
	int i, j, k;
	for (i = 0; i < MAT_SIZE; i += block_size){
		for (j = 0; j < MAT_SIZE; j += block_size){
			for (k = 0; k < MAT_SIZE; k += block_size)
				for(int i1 = i; i1 < i + block_size; i1++)
					for(int j1 = j; j1 < j + block_size; j1++){
						uint64_t temp = 0;
						for(int k1 = k; k1 + unrolling_factor5 - 1 < k + block_size; k1 += unrolling_factor5){
							temp += (matrix_A[i1 * MAT_SIZE + k1] * matrix_B[k1 * MAT_SIZE + j1]);
							temp += (matrix_A[i1 * MAT_SIZE + k1+1] * matrix_B[(k1+1) * MAT_SIZE + j1]);
							temp += (matrix_A[i1 * MAT_SIZE + k1+2] * matrix_B[(k1+2) * MAT_SIZE + j1]);
							temp += (matrix_A[i1 * MAT_SIZE + k1+3] * matrix_B[(k1+3) * MAT_SIZE + j1]);
							temp += (matrix_A[i1 * MAT_SIZE + k1+4] * matrix_B[(k1+4) * MAT_SIZE + j1]);
							temp += (matrix_A[i1 * MAT_SIZE + k1+5] * matrix_B[(k1+5) * MAT_SIZE + j1]);
							temp += (matrix_A[i1 * MAT_SIZE + k1+6] * matrix_B[(k1+6) * MAT_SIZE + j1]);
							temp += (matrix_A[i1 * MAT_SIZE + k1+7] * matrix_B[(k1+7) * MAT_SIZE + j1]);
							temp += (matrix_A[i1 * MAT_SIZE + k1+8] * matrix_B[(k1+8) * MAT_SIZE + j1]);
							temp += (matrix_A[i1 * MAT_SIZE + k1+9] * matrix_B[(k1+9) * MAT_SIZE + j1]);
							temp += (matrix_A[i1 * MAT_SIZE + k1+10] * matrix_B[(k1+10) * MAT_SIZE + j1]);
							temp += (matrix_A[i1 * MAT_SIZE + k1+11] * matrix_B[(k1+11) * MAT_SIZE + j1]);
							temp += (matrix_A[i1 * MAT_SIZE + k1+12] * matrix_B[(k1+12) * MAT_SIZE + j1]);
							temp += (matrix_A[i1 * MAT_SIZE + k1+13] * matrix_B[(k1+13) * MAT_SIZE + j1]);
							temp += (matrix_A[i1 * MAT_SIZE + k1+14] * matrix_B[(k1+14) * MAT_SIZE + j1]);
							temp += (matrix_A[i1 * MAT_SIZE + k1+15] * matrix_B[(k1+15) * MAT_SIZE + j1]);
							temp += (matrix_A[i1 * MAT_SIZE + k1+16] * matrix_B[(k1+16) * MAT_SIZE + j1]);
							temp += (matrix_A[i1 * MAT_SIZE + k1+17] * matrix_B[(k1+17) * MAT_SIZE + j1]);
							temp += (matrix_A[i1 * MAT_SIZE + k1+18] * matrix_B[(k1+18) * MAT_SIZE + j1]);
							temp += (matrix_A[i1 * MAT_SIZE + k1+19] * matrix_B[(k1+19) * MAT_SIZE + j1]);
							temp += (matrix_A[i1 * MAT_SIZE + k1+20] * matrix_B[(k1+20) * MAT_SIZE + j1]);
							temp += (matrix_A[i1 * MAT_SIZE + k1+21] * matrix_B[(k1+21) * MAT_SIZE + j1]);
							temp += (matrix_A[i1 * MAT_SIZE + k1+22] * matrix_B[(k1+22) * MAT_SIZE + j1]);
							temp += (matrix_A[i1 * MAT_SIZE + k1+23] * matrix_B[(k1+23) * MAT_SIZE + j1]);
							temp += (matrix_A[i1 * MAT_SIZE + k1+24] * matrix_B[(k1+24) * MAT_SIZE + j1]);
							temp += (matrix_A[i1 * MAT_SIZE + k1+25] * matrix_B[(k1+25) * MAT_SIZE + j1]);
							temp += (matrix_A[i1 * MAT_SIZE + k1+26] * matrix_B[(k1+26) * MAT_SIZE + j1]);
							temp += (matrix_A[i1 * MAT_SIZE + k1+27] * matrix_B[(k1+27) * MAT_SIZE + j1]);
							temp += (matrix_A[i1 * MAT_SIZE + k1+28] * matrix_B[(k1+28) * MAT_SIZE + j1]);
							temp += (matrix_A[i1 * MAT_SIZE + k1+29] * matrix_B[(k1+29) * MAT_SIZE + j1]);
							temp += (matrix_A[i1 * MAT_SIZE + k1+30] * matrix_B[(k1+30) * MAT_SIZE + j1]);
							temp += (matrix_A[i1 * MAT_SIZE + k1+31] * matrix_B[(k1+31) * MAT_SIZE + j1]);
						}
						for(int k1 = k + unrolling_factor5 * (block_size/unrolling_factor5); k1 < k + block_size; k1++){
							temp += (matrix_A[i1 * MAT_SIZE + k1] * matrix_B[k1 * MAT_SIZE + j1]);
						}
						sequential_opt_C_5[i1 * MAT_SIZE + j1] += temp;
					}
		}
	}
}

void sequential_matmul_opt_6() {
  // TODO:
	int i, j, k;
	for (i = 0; i < MAT_SIZE; i++){
		for (j = 0; j < MAT_SIZE; j += block_size){
			for (k = 0; k < MAT_SIZE; k += block_size)
				for(int j1 = j; j1 < j + block_size; j1++){
					uint64_t temp = 0;
					for(int k1 = k; k1 + unrolling_factor5 - 1 < k + block_size; k1 += unrolling_factor5){
						temp += (matrix_A[i * MAT_SIZE + k1] * matrix_B[k1 * MAT_SIZE + j1]);
							temp += (matrix_A[i * MAT_SIZE + k1+1] * matrix_B[(k1+1) * MAT_SIZE + j1]);
							temp += (matrix_A[i * MAT_SIZE + k1+2] * matrix_B[(k1+2) * MAT_SIZE + j1]);
							temp += (matrix_A[i * MAT_SIZE + k1+3] * matrix_B[(k1+3) * MAT_SIZE + j1]);
							temp += (matrix_A[i * MAT_SIZE + k1+4] * matrix_B[(k1+4) * MAT_SIZE + j1]);
							temp += (matrix_A[i * MAT_SIZE + k1+5] * matrix_B[(k1+5) * MAT_SIZE + j1]);
							temp += (matrix_A[i * MAT_SIZE + k1+6] * matrix_B[(k1+6) * MAT_SIZE + j1]);
							temp += (matrix_A[i * MAT_SIZE + k1+7] * matrix_B[(k1+7) * MAT_SIZE + j1]);
							temp += (matrix_A[i * MAT_SIZE + k1+8] * matrix_B[(k1+8) * MAT_SIZE + j1]);
							temp += (matrix_A[i * MAT_SIZE + k1+9] * matrix_B[(k1+9) * MAT_SIZE + j1]);
							temp += (matrix_A[i * MAT_SIZE + k1+10] * matrix_B[(k1+10) * MAT_SIZE + j1]);
							temp += (matrix_A[i * MAT_SIZE + k1+11] * matrix_B[(k1+11) * MAT_SIZE + j1]);
							temp += (matrix_A[i * MAT_SIZE + k1+12] * matrix_B[(k1+12) * MAT_SIZE + j1]);
							temp += (matrix_A[i * MAT_SIZE + k1+13] * matrix_B[(k1+13) * MAT_SIZE + j1]);
							temp += (matrix_A[i * MAT_SIZE + k1+14] * matrix_B[(k1+14) * MAT_SIZE + j1]);
							temp += (matrix_A[i * MAT_SIZE + k1+15] * matrix_B[(k1+15) * MAT_SIZE + j1]);
							temp += (matrix_A[i * MAT_SIZE + k1+16] * matrix_B[(k1+16) * MAT_SIZE + j1]);
							temp += (matrix_A[i * MAT_SIZE + k1+17] * matrix_B[(k1+17) * MAT_SIZE + j1]);
							temp += (matrix_A[i * MAT_SIZE + k1+18] * matrix_B[(k1+18) * MAT_SIZE + j1]);
							temp += (matrix_A[i * MAT_SIZE + k1+19] * matrix_B[(k1+19) * MAT_SIZE + j1]);
							temp += (matrix_A[i * MAT_SIZE + k1+20] * matrix_B[(k1+20) * MAT_SIZE + j1]);
							temp += (matrix_A[i * MAT_SIZE + k1+21] * matrix_B[(k1+21) * MAT_SIZE + j1]);
							temp += (matrix_A[i * MAT_SIZE + k1+22] * matrix_B[(k1+22) * MAT_SIZE + j1]);
							temp += (matrix_A[i * MAT_SIZE + k1+23] * matrix_B[(k1+23) * MAT_SIZE + j1]);
							temp += (matrix_A[i * MAT_SIZE + k1+24] * matrix_B[(k1+24) * MAT_SIZE + j1]);
							temp += (matrix_A[i * MAT_SIZE + k1+25] * matrix_B[(k1+25) * MAT_SIZE + j1]);
							temp += (matrix_A[i * MAT_SIZE + k1+26] * matrix_B[(k1+26) * MAT_SIZE + j1]);
							temp += (matrix_A[i * MAT_SIZE + k1+27] * matrix_B[(k1+27) * MAT_SIZE + j1]);
							temp += (matrix_A[i * MAT_SIZE + k1+28] * matrix_B[(k1+28) * MAT_SIZE + j1]);
							temp += (matrix_A[i * MAT_SIZE + k1+29] * matrix_B[(k1+29) * MAT_SIZE + j1]);
							temp += (matrix_A[i * MAT_SIZE + k1+30] * matrix_B[(k1+30) * MAT_SIZE + j1]);
							temp += (matrix_A[i * MAT_SIZE + k1+31] * matrix_B[(k1+31) * MAT_SIZE + j1]);
					}
					for(int k1 = k + unrolling_factor5 * (block_size/unrolling_factor5); k1 < k + block_size; k1++){
						temp += (matrix_A[i * MAT_SIZE + k1] * matrix_B[k1 * MAT_SIZE + j1]);
					}
					sequential_opt_C_6[i * MAT_SIZE + j1] += temp;
				}
		}
	}
}

void* parallel_matmul_opt(void* thread_id){
	int id = (intptr_t)thread_id;
	int i, j, k;
	for (i = id*MAT_SIZE/NUM_THREADS; i < (id+1)*MAT_SIZE/NUM_THREADS; i += block_size){
		for (j = 0; j < MAT_SIZE; j += block_size){
			for (k = 0; k < MAT_SIZE; k += block_size)
				for(int i1 = i; i1 < i + block_size; i1++)
					for(int j1 = j; j1 < j + block_size; j1++){
						uint64_t temp = 0;
						for(int k1 = k; k1 + unrolling_factor5 - 1 < k + block_size; k1 += unrolling_factor5){
							temp += (matrix_A[i1 * MAT_SIZE + k1] * matrix_B[k1 * MAT_SIZE + j1]);
							temp += (matrix_A[i1 * MAT_SIZE + k1+1] * matrix_B[(k1+1) * MAT_SIZE + j1]);
							temp += (matrix_A[i1 * MAT_SIZE + k1+2] * matrix_B[(k1+2) * MAT_SIZE + j1]);
							temp += (matrix_A[i1 * MAT_SIZE + k1+3] * matrix_B[(k1+3) * MAT_SIZE + j1]);
							temp += (matrix_A[i1 * MAT_SIZE + k1+4] * matrix_B[(k1+4) * MAT_SIZE + j1]);
							temp += (matrix_A[i1 * MAT_SIZE + k1+5] * matrix_B[(k1+5) * MAT_SIZE + j1]);
							temp += (matrix_A[i1 * MAT_SIZE + k1+6] * matrix_B[(k1+6) * MAT_SIZE + j1]);
							temp += (matrix_A[i1 * MAT_SIZE + k1+7] * matrix_B[(k1+7) * MAT_SIZE + j1]);
							temp += (matrix_A[i1 * MAT_SIZE + k1+8] * matrix_B[(k1+8) * MAT_SIZE + j1]);
							temp += (matrix_A[i1 * MAT_SIZE + k1+9] * matrix_B[(k1+9) * MAT_SIZE + j1]);
							temp += (matrix_A[i1 * MAT_SIZE + k1+10] * matrix_B[(k1+10) * MAT_SIZE + j1]);
							temp += (matrix_A[i1 * MAT_SIZE + k1+11] * matrix_B[(k1+11) * MAT_SIZE + j1]);
							temp += (matrix_A[i1 * MAT_SIZE + k1+12] * matrix_B[(k1+12) * MAT_SIZE + j1]);
							temp += (matrix_A[i1 * MAT_SIZE + k1+13] * matrix_B[(k1+13) * MAT_SIZE + j1]);
							temp += (matrix_A[i1 * MAT_SIZE + k1+14] * matrix_B[(k1+14) * MAT_SIZE + j1]);
							temp += (matrix_A[i1 * MAT_SIZE + k1+15] * matrix_B[(k1+15) * MAT_SIZE + j1]);
							temp += (matrix_A[i1 * MAT_SIZE + k1+16] * matrix_B[(k1+16) * MAT_SIZE + j1]);
							temp += (matrix_A[i1 * MAT_SIZE + k1+17] * matrix_B[(k1+17) * MAT_SIZE + j1]);
							temp += (matrix_A[i1 * MAT_SIZE + k1+18] * matrix_B[(k1+18) * MAT_SIZE + j1]);
							temp += (matrix_A[i1 * MAT_SIZE + k1+19] * matrix_B[(k1+19) * MAT_SIZE + j1]);
							temp += (matrix_A[i1 * MAT_SIZE + k1+20] * matrix_B[(k1+20) * MAT_SIZE + j1]);
							temp += (matrix_A[i1 * MAT_SIZE + k1+21] * matrix_B[(k1+21) * MAT_SIZE + j1]);
							temp += (matrix_A[i1 * MAT_SIZE + k1+22] * matrix_B[(k1+22) * MAT_SIZE + j1]);
							temp += (matrix_A[i1 * MAT_SIZE + k1+23] * matrix_B[(k1+23) * MAT_SIZE + j1]);
							temp += (matrix_A[i1 * MAT_SIZE + k1+24] * matrix_B[(k1+24) * MAT_SIZE + j1]);
							temp += (matrix_A[i1 * MAT_SIZE + k1+25] * matrix_B[(k1+25) * MAT_SIZE + j1]);
							temp += (matrix_A[i1 * MAT_SIZE + k1+26] * matrix_B[(k1+26) * MAT_SIZE + j1]);
							temp += (matrix_A[i1 * MAT_SIZE + k1+27] * matrix_B[(k1+27) * MAT_SIZE + j1]);
							temp += (matrix_A[i1 * MAT_SIZE + k1+28] * matrix_B[(k1+28) * MAT_SIZE + j1]);
							temp += (matrix_A[i1 * MAT_SIZE + k1+29] * matrix_B[(k1+29) * MAT_SIZE + j1]);
							temp += (matrix_A[i1 * MAT_SIZE + k1+30] * matrix_B[(k1+30) * MAT_SIZE + j1]);
							temp += (matrix_A[i1 * MAT_SIZE + k1+31] * matrix_B[(k1+31) * MAT_SIZE + j1]);
						}
						for(int k1 = k + unrolling_factor5 * (block_size/unrolling_factor5); k1 < k + block_size; k1++){
							temp += (matrix_A[i1 * MAT_SIZE + k1] * matrix_B[k1 * MAT_SIZE + j1]);
						}
						parallel_opt_C[i1 * MAT_SIZE + j1] += temp;
					}
		}
	}
    pthread_exit(NULL);
}

double rtclock() {
	struct timezone Tzp;
	struct timeval Tp;
	int stat;
	stat = gettimeofday(&Tp, &Tzp);
	if (stat != 0){
		printf("Error return from gettimeofday: %d\n", stat);
	}
	return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

void check_result(uint64_t* first_res, uint64_t* second_res) {
	double maxdiff, this_diff;
	int numdiffs;
	int i, j;
	numdiffs = 0;
	maxdiff = 0;

	for (i = 0; i < MAT_SIZE; i++) {
		for (j = 0; j < MAT_SIZE; j++) {
		this_diff = first_res[i * MAT_SIZE + j] - second_res[i * MAT_SIZE + j];
		if (this_diff < 0)
			this_diff = -1.0 * this_diff;
		if (this_diff > THRESHOLD) {
			numdiffs++;
			if (this_diff > maxdiff)
			maxdiff = this_diff;
		}
		}
	}

	if (numdiffs > 0) {
		cout << numdiffs << " Diffs found over threshold " << THRESHOLD << "; Max Diff = " << maxdiff
			<< endl;
	} else {
		cout << "No differences found between base and test versions\n";
	}
}

int main(int argc, char* argv[]) {
	if (argc == 3) {
		block_size = atoi(argv[1]);
		NUM_THREADS = atoi(argv[2]);
	}
	else if(argc == 2){
		block_size = atoi(argv[1]);
		NUM_THREADS = 1;
		cout << "Using default num of threads = 1\n";
	} 
	else {
		block_size = 4;
		NUM_THREADS = 1;
		cout << "Using default block size = 4 and num of threads = 1\n";
	}
	matrix_A = new (std::nothrow) uint64_t[MAT_SIZE * MAT_SIZE];
	matrix_B = new (std::nothrow) uint64_t[MAT_SIZE * MAT_SIZE];
	sequential_C = new (std::nothrow) uint64_t[MAT_SIZE * MAT_SIZE];
	parallel_C = new (std::nothrow) uint64_t[MAT_SIZE * MAT_SIZE];
	sequential_opt_C_1 = new (std::nothrow) uint64_t[MAT_SIZE * MAT_SIZE];
	sequential_opt_C_2 = new (std::nothrow) uint64_t[MAT_SIZE * MAT_SIZE];
	sequential_opt_C_3 = new (std::nothrow) uint64_t[MAT_SIZE * MAT_SIZE];
	sequential_opt_C_4 = new (std::nothrow) uint64_t[MAT_SIZE * MAT_SIZE];
	sequential_opt_C_5 = new (std::nothrow) uint64_t[MAT_SIZE * MAT_SIZE];
	sequential_opt_C_6 = new (std::nothrow) uint64_t[MAT_SIZE * MAT_SIZE];
	parallel_opt_C = new (std::nothrow) uint64_t[MAT_SIZE * MAT_SIZE];

	for (int i = 0; i < MAT_SIZE; i++) {
		for (int j = 0; j < MAT_SIZE; j++) {
		matrix_A[(i * MAT_SIZE) + j] = 1;
		matrix_B[(i * MAT_SIZE) + j] = 1;
		sequential_C[(i * MAT_SIZE) + j] = 0;
		parallel_C[(i * MAT_SIZE) + j] = 0;
		sequential_opt_C_1[(i * MAT_SIZE) + j] = 0;
		sequential_opt_C_2[(i * MAT_SIZE) + j] = 0;
		sequential_opt_C_3[(i * MAT_SIZE) + j] = 0;
		sequential_opt_C_4[(i * MAT_SIZE) + j] = 0;
		sequential_opt_C_5[(i * MAT_SIZE) + j] = 0;
		sequential_opt_C_6[(i * MAT_SIZE) + j] = 0;
		parallel_opt_C[(i * MAT_SIZE) + j] = 0;
		}
	}
	pthread_t thread_arr[NUM_THREADS];
	pthread_attr_t attr;

	double clkbegin, clkend;

	clkbegin = rtclock();
	sequential_matmul();
	clkend = rtclock();
	cout << "Time for Sequential version: " << (clkend - clkbegin) << "seconds.\n";

	clkbegin = rtclock();
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

	for (int i = 0; i < NUM_THREADS; i++) {
		pthread_create(&thread_arr[i], &attr, parallel_matmul, (void*)(intptr_t)i);
	}

	for (int i = 0; i < NUM_THREADS; i++) {
		pthread_join(thread_arr[i], NULL);
	}

	clkend = rtclock();
	cout << "Time for Parallel version: " << (clkend - clkbegin) << "seconds.\n";

	// clkbegin = rtclock();
	// sequential_matmul_opt_1();
	// clkend = rtclock();
	// cout << "Time for Sequential Optimized 1 version: " << (clkend - clkbegin) << "seconds.\n";

	// clkbegin = rtclock();
	// sequential_matmul_opt_2();
	// clkend = rtclock();
	// cout << "Time for Sequential Optimized 2 version: " << (clkend - clkbegin) << "seconds.\n";

	// clkbegin = rtclock();
	// sequential_matmul_opt_3();
	// clkend = rtclock();
	// cout << "Time for Sequential Optimized 3 version: " << (clkend - clkbegin) << "seconds.\n";

	// clkbegin = rtclock();
	// sequential_matmul_opt_4();
	// clkend = rtclock();
	// cout << "Time for Sequential Optimized 4 version: " << (clkend - clkbegin) << "seconds.\n";

	clkbegin = rtclock();
	sequential_matmul_opt_5();
	clkend = rtclock();
	cout << "Time for Sequential Optimized 5 version: " << (clkend - clkbegin) << "seconds.\n";

	// clkbegin = rtclock();
	// sequential_matmul_opt_6();
	// clkend = rtclock();
	// cout << "Time for Sequential Optimized 6 version: " << (clkend - clkbegin) << "seconds.\n";

	clkbegin = rtclock();

	for (int i = 0; i < NUM_THREADS; i++) {
		pthread_create(&thread_arr[i], &attr, parallel_matmul_opt, (void*)(intptr_t)i);
	}

	for (int i = 0; i < NUM_THREADS; i++) {
		pthread_join(thread_arr[i], NULL);
	}

	clkend = rtclock();
	cout << "Time for Parallel Optimized version: " << (clkend - clkbegin) << "seconds.\n";

	check_result(sequential_C, parallel_C);
	// check_result(sequential_C, sequential_opt_C_1);
	// check_result(sequential_C, sequential_opt_C_2);
	// check_result(sequential_C, sequential_opt_C_3);
	// check_result(sequential_C, sequential_opt_C_4);
	check_result(sequential_C, sequential_opt_C_5);
	// check_result(sequential_C, sequential_opt_C_6);
	check_result(sequential_C, parallel_opt_C);

    pthread_attr_destroy(&attr);
	pthread_exit(NULL);

	return 0;
}
