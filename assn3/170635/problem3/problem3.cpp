// Compile: g++ -O2 -fopenmp -o problem3 problem3.cpp
// Execute: ./problem3 <num_threads> <block_size>

#include <cassert>
#include <iostream>
#include <omp.h>

#define N (1 << 12)
#define ITER 100

using namespace std;

int num_threads; // taken from command line arg
const int chunk_size = 32;
int block_size; // taken from command line arg

void check_result(uint32_t** w_ref, uint32_t** w_opt) {
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			assert(w_ref[i][j] == w_opt[i][j]);
		}
	}
	cout << "No differences found between base and test versions\n";
}

void reset(uint32_t** A_omp){
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			A_omp[i][j] = i + j + 1;
		}
	}
}

void reference(uint32_t** A) {
	int i, j, k;
	for (k = 0; k < ITER; k++) {
		for (i = 1; i < N; i++) {
			for (j = 0; j < (N - 1); j++) {
				A[i][j + 1] = A[i - 1][j + 1] + A[i][j + 1];
			}
		}
	}
}

// TODO: MAKE YOUR CHANGES TO OPTIMIZE THIS FUNCTION
void omp_version1(uint32_t** A) {
	int i, j, k;
	#pragma omp parallel for num_threads(num_threads) default(none) private(k,i) shared(A) schedule(dynamic, chunk_size)
	for (j = 0; j < (N - 1); j++) {
		for (k = 0; k < ITER; k++) {
			for (i = 1; i < N; i++) {
				A[i][j + 1] = A[i - 1][j + 1] + A[i][j + 1];
			}
		}
	}
}

void omp_version2(uint32_t** A) {
	int i, j, k;
	for (k = 0; k < ITER; k++) {
		for (i = 1; i < N; i++) {
			#pragma omp parallel for num_threads(num_threads) schedule(dynamic, chunk_size)
			for (j = 0; j < (N - 1); j++) {
				A[i][j + 1] = A[i - 1][j + 1] + A[i][j + 1];
			}
		}
	}
}

void omp_version3(uint32_t** A) {
	int i, j, k;
	for (k = 0; k < ITER; k++) {
		#pragma omp parallel for num_threads(num_threads) default(none) private(i) shared(A) schedule(dynamic, chunk_size)
		for (j = 0; j < (N - 1); j++) {
			for (i = 1; i < N; i++) {
				A[i][j + 1] = A[i - 1][j + 1] + A[i][j + 1];
			}
		}
	}
}

void omp_version4(uint32_t** A) {
	int i, j, k;
	#pragma omp parallel for num_threads(num_threads) default(none) private(k,i) shared(A, block_size)
	for (j = 0; j < (N - 1); j += block_size) {
		for (k = 0; k < ITER; k++) {
			for (i = 1; i < N; i += block_size) {
				int limit1 = min(j+block_size, N-1);
				for(int j1 = j; j1 < limit1; j1++){
					int limit2 = min(i+block_size, N);
					for(int i1 = i; i1 < limit2; i1++){
						A[i1][j1 + 1] = A[i1 - 1][j1 + 1] + A[i1][j1 + 1];
					}
				}
			}
		}
	}
}

void omp_version5(uint32_t** A) {
	int i, j, k;
	for (k = 0; k < ITER; k++) {
		for (i = 1; i < N; i += block_size) {
			#pragma omp parallel for num_threads(num_threads)
			for (j = 0; j < (N - 1); j += block_size) {
				int limit1 = min(i+block_size, N);
				for(int i1 = i; i1 < limit1; i1++){
					int limit2 = min(j+block_size, N-1);
					for(int j1 = j; j1 < limit2; j1++){
						A[i1][j1 + 1] = A[i1 - 1][j1 + 1] + A[i1][j1 + 1];
					}
				}
			}
		}
	}
}

void omp_version6(uint32_t** A) {
	int i, j, k;
	for (k = 0; k < ITER; k++) {
		#pragma omp parallel for num_threads(num_threads) default(none) private(i) shared(A, block_size)
		for (j = 0; j < (N - 1); j += block_size) {
			for (i = 1; i < N; i += block_size) {
				int limit1 = min(j+block_size, N-1);
				for(int j1 = j; j1 < limit1; j1++){
					int limit2 = min(i+block_size, N);
					for(int i1 = i; i1 < limit2; i1++){
						A[i1][j1 + 1] = A[i1 - 1][j1 + 1] + A[i1][j1 + 1];
					}
				}
			}
		}
	}
}

void omp_version7(uint32_t** A) {
	int i, j, k;
	int unrolling_factor = 2;
	#pragma omp parallel for num_threads(num_threads) default(none) private(k,i) shared(A, block_size, unrolling_factor)
	for (j = 0; j < (N - 1); j += block_size) {
		for (k = 0; k < ITER; k++) {
			for (i = 1; i < N; i += block_size) {
				int limit1 = min(j+block_size, N-1);
				for(int j1 = j; j1 < limit1; j1++){
					int limit2 = min(i+block_size, N);
					for(int i1 = i; i1 + unrolling_factor - 1 < limit2; i1 += unrolling_factor){
						A[i1][j1 + 1] = A[i1 - 1][j1 + 1] + A[i1][j1 + 1];
						A[i1+1][j1 + 1] = A[i1][j1 + 1] + A[i1+1][j1 + 1];
					}
					for(int i1 = i + unrolling_factor*((limit2-i)/unrolling_factor); i1 < limit2; i1++){
						A[i1][j1 + 1] = A[i1 - 1][j1 + 1] + A[i1][j1 + 1];
					}
				}
			}
		}
	}
}

void omp_version8(uint32_t** A) {
	int i, j, k;
	int unrolling_factor = 4;
	#pragma omp parallel for num_threads(num_threads) default(none) private(k,i) shared(A, block_size, unrolling_factor)
	for (j = 0; j < (N - 1); j += block_size) {
		for (k = 0; k < ITER; k++) {
			for (i = 1; i < N; i += block_size) {
				int limit1 = min(j+block_size, N-1);
				for(int j1 = j; j1 < limit1; j1++){
					int limit2 = min(i+block_size, N);
					for(int i1 = i; i1 + unrolling_factor - 1 < limit2; i1 += unrolling_factor){
						A[i1][j1 + 1] = A[i1 - 1][j1 + 1] + A[i1][j1 + 1];
						A[i1+1][j1 + 1] = A[i1][j1 + 1] + A[i1+1][j1 + 1];
						A[i1+2][j1 + 1] = A[i1+1][j1 + 1] + A[i1+2][j1 + 1];
						A[i1+3][j1 + 1] = A[i1+2][j1 + 1] + A[i1+3][j1 + 1];
					}
					for(int i1 = i + unrolling_factor*((limit2-i)/unrolling_factor); i1 < limit2; i1++){
						A[i1][j1 + 1] = A[i1 - 1][j1 + 1] + A[i1][j1 + 1];
					}
				}
			}
		}
	}
}

void omp_version9(uint32_t** A) {
	int i, j, k;
	int unrolling_factor = 8;
	#pragma omp parallel for num_threads(num_threads) default(none) private(k,i) shared(A, block_size, unrolling_factor)
	for (j = 0; j < (N - 1); j += block_size) {
		for (k = 0; k < ITER; k++) {
			for (i = 1; i < N; i += block_size) {
				int limit1 = min(j+block_size, N-1);
				for(int j1 = j; j1 < limit1; j1++){
					int limit2 = min(i+block_size, N);
					for(int i1 = i; i1 + unrolling_factor - 1 < limit2; i1 += unrolling_factor){
						A[i1][j1 + 1] = A[i1 - 1][j1 + 1] + A[i1][j1 + 1];
						A[i1+1][j1 + 1] = A[i1][j1 + 1] + A[i1+1][j1 + 1];
						A[i1+2][j1 + 1] = A[i1+1][j1 + 1] + A[i1+2][j1 + 1];
						A[i1+3][j1 + 1] = A[i1+2][j1 + 1] + A[i1+3][j1 + 1];
						A[i1+4][j1 + 1] = A[i1+3][j1 + 1] + A[i1+4][j1 + 1];
						A[i1+5][j1 + 1] = A[i1+4][j1 + 1] + A[i1+5][j1 + 1];
						A[i1+6][j1 + 1] = A[i1+5][j1 + 1] + A[i1+6][j1 + 1];
						A[i1+7][j1 + 1] = A[i1+6][j1 + 1] + A[i1+7][j1 + 1];
					}
					for(int i1 = i + unrolling_factor*((limit2-i)/unrolling_factor); i1 < limit2; i1++){
						A[i1][j1 + 1] = A[i1 - 1][j1 + 1] + A[i1][j1 + 1];
					}
				}
			}
		}
	}
}

void omp_version10(uint32_t** A) {
	int i, j, k;
	int unrolling_factor = 16;
	#pragma omp parallel for num_threads(num_threads) default(none) private(k,i) shared(A, block_size, unrolling_factor)
	for (j = 0; j < (N - 1); j += block_size) {
		for (k = 0; k < ITER; k++) {
			for (i = 1; i < N; i += block_size) {
				int limit1 = min(j+block_size, N-1);
				for(int j1 = j; j1 < limit1; j1++){
					int limit2 = min(i+block_size, N);
					for(int i1 = i; i1 + unrolling_factor - 1 < limit2; i1 += unrolling_factor){
						A[i1][j1 + 1] = A[i1 - 1][j1 + 1] + A[i1][j1 + 1];
						A[i1+1][j1 + 1] = A[i1][j1 + 1] + A[i1+1][j1 + 1];
						A[i1+2][j1 + 1] = A[i1+1][j1 + 1] + A[i1+2][j1 + 1];
						A[i1+3][j1 + 1] = A[i1+2][j1 + 1] + A[i1+3][j1 + 1];
						A[i1+4][j1 + 1] = A[i1+3][j1 + 1] + A[i1+4][j1 + 1];
						A[i1+5][j1 + 1] = A[i1+4][j1 + 1] + A[i1+5][j1 + 1];
						A[i1+6][j1 + 1] = A[i1+5][j1 + 1] + A[i1+6][j1 + 1];
						A[i1+7][j1 + 1] = A[i1+6][j1 + 1] + A[i1+7][j1 + 1];
						A[i1+8][j1 + 1] = A[i1+7][j1 + 1] + A[i1+8][j1 + 1];
						A[i1+9][j1 + 1] = A[i1+8][j1 + 1] + A[i1+9][j1 + 1];
						A[i1+10][j1 + 1] = A[i1+9][j1 + 1] + A[i1+10][j1 + 1];
						A[i1+11][j1 + 1] = A[i1+10][j1 + 1] + A[i1+11][j1 + 1];
						A[i1+12][j1 + 1] = A[i1+11][j1 + 1] + A[i1+12][j1 + 1];
						A[i1+13][j1 + 1] = A[i1+12][j1 + 1] + A[i1+13][j1 + 1];
						A[i1+14][j1 + 1] = A[i1+13][j1 + 1] + A[i1+14][j1 + 1];
						A[i1+15][j1 + 1] = A[i1+14][j1 + 1] + A[i1+15][j1 + 1];
					}
					for(int i1 = i + unrolling_factor*((limit2-i)/unrolling_factor); i1 < limit2; i1++){
						A[i1][j1 + 1] = A[i1 - 1][j1 + 1] + A[i1][j1 + 1];
					}
				}
			}
		}
	}
}

void omp_version11(uint32_t** A) {
	int i, j, k;
	int unrolling_factor = 2;
	#pragma omp parallel for num_threads(num_threads) default(none) private(k,i) shared(A, block_size, unrolling_factor)
	for (j = 0; j < (N - 1); j += block_size) {
		for (k = 0; k < ITER; k++) {
			for (i = 1; i < N; i += block_size) {
				int limit1 = min(j+block_size, N-1);
				for(int j1 = j; j1 + unrolling_factor - 1 < limit1; j1 += unrolling_factor){
					int limit2 = min(i+block_size, N);
					for(int i1 = i; i1 < limit2; i1++){
						A[i1][j1 + 1] = A[i1 - 1][j1 + 1] + A[i1][j1 + 1];
						A[i1][j1 + 2] = A[i1 - 1][j1 + 2] + A[i1][j1 + 2];
					}
				}
				for(int j1 = j + unrolling_factor*((limit1-j)/unrolling_factor); j1 < limit1; j1++){
					int limit2 = min(i+block_size, N);
					for(int i1 = i; i1 < limit2; i1++){
						A[i1][j1 + 1] = A[i1 - 1][j1 + 1] + A[i1][j1 + 1];
					}
				}
			}
		}
	}
}

void omp_version12(uint32_t** A) {
	int i, j, k;
	int unrolling_factor = 4;
	#pragma omp parallel for num_threads(num_threads) default(none) private(k,i) shared(A, block_size, unrolling_factor)
	for (j = 0; j < (N - 1); j += block_size) {
		for (k = 0; k < ITER; k++) {
			for (i = 1; i < N; i += block_size) {
				int limit1 = min(j+block_size, N-1);
				for(int j1 = j; j1 + unrolling_factor - 1 < limit1; j1 += unrolling_factor){
					int limit2 = min(i+block_size, N);
					for(int i1 = i; i1 < limit2; i1++){
						A[i1][j1 + 1] = A[i1 - 1][j1 + 1] + A[i1][j1 + 1];
						A[i1][j1 + 2] = A[i1 - 1][j1 + 2] + A[i1][j1 + 2];
						A[i1][j1 + 3] = A[i1 - 1][j1 + 3] + A[i1][j1 + 3];
						A[i1][j1 + 4] = A[i1 - 1][j1 + 4] + A[i1][j1 + 4];
					}
				}
				for(int j1 = j + unrolling_factor*((limit1-j)/unrolling_factor); j1 < limit1; j1++){
					int limit2 = min(i+block_size, N);
					for(int i1 = i; i1 < limit2; i1++){
						A[i1][j1 + 1] = A[i1 - 1][j1 + 1] + A[i1][j1 + 1];
					}
				}
			}
		}
	}
}

void omp_version13(uint32_t** A) {
	int i, j, k;
	int unrolling_factor = 8;
	#pragma omp parallel for num_threads(num_threads) default(none) private(k,i) shared(A, block_size, unrolling_factor)
	for (j = 0; j < (N - 1); j += block_size) {
		for (k = 0; k < ITER; k++) {
			for (i = 1; i < N; i += block_size) {
				int limit1 = min(j+block_size, N-1);
				for(int j1 = j; j1 + unrolling_factor - 1 < limit1; j1 += unrolling_factor){
					int limit2 = min(i+block_size, N);
					for(int i1 = i; i1 < limit2; i1++){
						A[i1][j1 + 1] = A[i1 - 1][j1 + 1] + A[i1][j1 + 1];
						A[i1][j1 + 2] = A[i1 - 1][j1 + 2] + A[i1][j1 + 2];
						A[i1][j1 + 3] = A[i1 - 1][j1 + 3] + A[i1][j1 + 3];
						A[i1][j1 + 4] = A[i1 - 1][j1 + 4] + A[i1][j1 + 4];
						A[i1][j1 + 5] = A[i1 - 1][j1 + 5] + A[i1][j1 + 5];
						A[i1][j1 + 6] = A[i1 - 1][j1 + 6] + A[i1][j1 + 6];
						A[i1][j1 + 7] = A[i1 - 1][j1 + 7] + A[i1][j1 + 7];
						A[i1][j1 + 8] = A[i1 - 1][j1 + 8] + A[i1][j1 + 8];
					}
				}
				for(int j1 = j + unrolling_factor*((limit1-j)/unrolling_factor); j1 < limit1; j1++){
					int limit2 = min(i+block_size, N);
					for(int i1 = i; i1 < limit2; i1++){
						A[i1][j1 + 1] = A[i1 - 1][j1 + 1] + A[i1][j1 + 1];
					}
				}
			}
		}
	}
}

void omp_version14(uint32_t** A) {
	int i, j, k;
	int unrolling_factor = 16;
	#pragma omp parallel for num_threads(num_threads) default(none) private(k,i) shared(A, block_size, unrolling_factor)
	for (j = 0; j < (N - 1); j += block_size) {
		for (k = 0; k < ITER; k++) {
			for (i = 1; i < N; i += block_size) {
				int limit1 = min(j+block_size, N-1);
				for(int j1 = j; j1 + unrolling_factor - 1 < limit1; j1 += unrolling_factor){
					int limit2 = min(i+block_size, N);
					for(int i1 = i; i1 < limit2; i1++){
						A[i1][j1 + 1] = A[i1 - 1][j1 + 1] + A[i1][j1 + 1];
						A[i1][j1 + 2] = A[i1 - 1][j1 + 2] + A[i1][j1 + 2];
						A[i1][j1 + 3] = A[i1 - 1][j1 + 3] + A[i1][j1 + 3];
						A[i1][j1 + 4] = A[i1 - 1][j1 + 4] + A[i1][j1 + 4];
						A[i1][j1 + 5] = A[i1 - 1][j1 + 5] + A[i1][j1 + 5];
						A[i1][j1 + 6] = A[i1 - 1][j1 + 6] + A[i1][j1 + 6];
						A[i1][j1 + 7] = A[i1 - 1][j1 + 7] + A[i1][j1 + 7];
						A[i1][j1 + 8] = A[i1 - 1][j1 + 8] + A[i1][j1 + 8];
						A[i1][j1 + 9] = A[i1 - 1][j1 + 9] + A[i1][j1 + 9];
						A[i1][j1 + 10] = A[i1 - 1][j1 + 10] + A[i1][j1 + 10];
						A[i1][j1 + 11] = A[i1 - 1][j1 + 11] + A[i1][j1 + 11];
						A[i1][j1 + 12] = A[i1 - 1][j1 + 12] + A[i1][j1 + 12];
						A[i1][j1 + 13] = A[i1 - 1][j1 + 13] + A[i1][j1 + 13];
						A[i1][j1 + 14] = A[i1 - 1][j1 + 14] + A[i1][j1 + 14];
						A[i1][j1 + 15] = A[i1 - 1][j1 + 15] + A[i1][j1 + 15];
						A[i1][j1 + 16] = A[i1 - 1][j1 + 16] + A[i1][j1 + 16];
					}
				}
				for(int j1 = j + unrolling_factor*((limit1-j)/unrolling_factor); j1 < limit1; j1++){
					int limit2 = min(i+block_size, N);
					for(int i1 = i; i1 < limit2; i1++){
						A[i1][j1 + 1] = A[i1 - 1][j1 + 1] + A[i1][j1 + 1];
					}
				}
			}
		}
	}
}

int main(int argc, char* argv[]) {
	if(argc == 3){
		num_threads = atoi(argv[1]);
		block_size = atoi(argv[2]);
	}
	else if(argc == 2){
		num_threads = atoi(argv[1]);
		block_size = 4;
		cout << "Using default block size = 4\n";
	}
	else{
		num_threads = 1;
		block_size = 4;
		cout << "Using default number of threads = 1 and block size = 4\n";
	}

	uint32_t** A_ref = new uint32_t*[N];
	for (int i = 0; i < N; i++) {
		A_ref[i] = new uint32_t[N];
	}

	uint32_t** A_omp = new uint32_t*[N];
	for (int i = 0; i < N; i++) {
		A_omp[i] = new uint32_t[N];
	}

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			A_ref[i][j] = i + j + 1;
			A_omp[i][j] = i + j + 1;
		}
	}

	double start = omp_get_wtime();
	reference(A_ref);
	double end = omp_get_wtime();
	cout << "Time for reference version: " << end - start << " seconds\n";

	// start = omp_get_wtime();
	// omp_version1(A_omp);
	// end = omp_get_wtime();
	// cout << "Version1: Time with OpenMP: " << end - start << " seconds\n";
	// check_result(A_ref, A_omp);
	// // Reset
	// reset(A_omp);

	// start = omp_get_wtime();
	// omp_version2(A_omp);
	// end = omp_get_wtime();
	// cout << "Version2: Time with OpenMP: " << end - start << " seconds\n";
	// check_result(A_ref, A_omp);
	// // Reset
	// reset(A_omp);

	// start = omp_get_wtime();
	// omp_version3(A_omp);
	// end = omp_get_wtime();
	// cout << "Version3: Time with OpenMP: " << end - start << " seconds\n";
	// check_result(A_ref, A_omp);
	// // Reset
	// reset(A_omp);

	// start = omp_get_wtime();
	// omp_version4(A_omp);
	// end = omp_get_wtime();
	// cout << "Version4: Time with OpenMP: " << end - start << " seconds\n";
	// check_result(A_ref, A_omp);
	// // Reset
	// reset(A_omp);

	// start = omp_get_wtime();
	// omp_version5(A_omp);
	// end = omp_get_wtime();
	// cout << "Version5: Time with OpenMP: " << end - start << " seconds\n";
	// check_result(A_ref, A_omp);
	// // Reset
	// reset(A_omp);

	// start = omp_get_wtime();
	// omp_version6(A_omp);
	// end = omp_get_wtime();
	// cout << "Version6: Time with OpenMP: " << end - start << " seconds\n";
	// check_result(A_ref, A_omp);
	// // Reset
	// reset(A_omp);

	// start = omp_get_wtime();
	// omp_version7(A_omp);
	// end = omp_get_wtime();
	// cout << "Version7: Time with OpenMP: " << end - start << " seconds\n";
	// check_result(A_ref, A_omp);
	// // Reset
	// reset(A_omp);

	// start = omp_get_wtime();
	// omp_version8(A_omp);
	// end = omp_get_wtime();
	// cout << "Version8: Time with OpenMP: " << end - start << " seconds\n";
	// check_result(A_ref, A_omp);
	// // Reset
	// reset(A_omp);

	// start = omp_get_wtime();
	// omp_version9(A_omp);
	// end = omp_get_wtime();
	// cout << "Version9: Time with OpenMP: " << end - start << " seconds\n";
	// check_result(A_ref, A_omp);
	// // Reset
	// reset(A_omp);

	start = omp_get_wtime();
	omp_version10(A_omp);
	end = omp_get_wtime();
	cout << "Version10: Time with OpenMP: " << end - start << " seconds\n";
	check_result(A_ref, A_omp);
	// Reset
	reset(A_omp);

	// start = omp_get_wtime();
	// omp_version11(A_omp);
	// end = omp_get_wtime();
	// cout << "Version11: Time with OpenMP: " << end - start << " seconds\n";
	// check_result(A_ref, A_omp);
	// // Reset
	// reset(A_omp);

	// start = omp_get_wtime();
	// omp_version12(A_omp);
	// end = omp_get_wtime();
	// cout << "Version12: Time with OpenMP: " << end - start << " seconds\n";
	// check_result(A_ref, A_omp);
	// // Reset
	// reset(A_omp);
	
	// start = omp_get_wtime();
	// omp_version13(A_omp);
	// end = omp_get_wtime();
	// cout << "Version13: Time with OpenMP: " << end - start << " seconds\n";
	// check_result(A_ref, A_omp);
	// // Reset
	// reset(A_omp);

	// start = omp_get_wtime();
	// omp_version14(A_omp);
	// end = omp_get_wtime();
	// cout << "Version14: Time with OpenMP: " << end - start << " seconds\n";
	// check_result(A_ref, A_omp);
	// // Reset
	// reset(A_omp);

	cout << '\n';

	return EXIT_SUCCESS;
}
