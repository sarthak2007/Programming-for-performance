// Compile: g++ -O2 -o problem2 problem2.cpp -mavx
// Execute: ./problem2 <block_size>

#include <cmath>
#include <iostream>
#include <sys/time.h>
#include <unistd.h>
#include <immintrin.h>

using namespace std;

const int N = 1024;
const int Niter = 10;
const double THRESHOLD = 0.0000001;

int block_size; // taken from command line arg
int unrolling_factor1 = 2;
int unrolling_factor2 = 4;
int unrolling_factor3 = 8;
int unrolling_factor4 = 16;

double rtclock() {
	struct timezone Tzp;
	struct timeval Tp;
	int stat;
	stat = gettimeofday(&Tp, &Tzp);
	if (stat != 0) {
		cout << "Error return from gettimeofday: " << stat << endl;
	}
	return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

void check_result(double** w_ref, double** w_opt) {
	double maxdiff, this_diff;
	int numdiffs;
	int i, j;
	numdiffs = 0;
	maxdiff = 0;

	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
		this_diff = w_ref[i][j] - w_opt[i][j];
		if (fabs(this_diff) > THRESHOLD) {
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

void reset(double** C_opt){
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			C_opt[i][j] = 0.0;
		}
	}
}

void reference(double** A, double** B, double** C) {
	int i, j, k;
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = 0; k < i + 1; k++) {
				C[i][j] += A[k][i] * B[j][k];
			}
		}
	}
}

// TODO: THIS IS INITIALLY IDENTICAL TO REFERENCE. MAKE YOUR CHANGES TO OPTIMIZE THIS FUNCTION
// You can create multiple versions of the optimized() function to test your changes
void optimized1(double** A, double** B, double** C) {
	int i, j, k;
	for (k = 0; k < N; k++) {
		for (i = k; i < N; i++) {
			for (j = 0; j < N; j++) {
				C[i][j] += A[k][i] * B[j][k];
			}
		}
	}
}

void optimized2(double** A, double** B, double** C) {
	int i, j, k;
	for (j = 0; j < N; j++) {
		for (k = 0; k < N; k++) {
			for (i = k; i < N; i++) {
				C[i][j] += A[k][i] * B[j][k];
			}
		}
	}
}

void optimized3(double** A, double** B, double** C) {
	int i, j, k;
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			double temp = 0;
			for (k = 0; k < i + 1; k++) {
				temp += A[k][i] * B[j][k];
			}
			C[i][j] += temp;
		}
	}
}

void optimized4(double** A, double** B, double** C) {
	int i, j, k;
	for (i = 0; i < N; i += block_size) {
		for (j = 0; j < N; j += block_size) {
			for (k = 0; k < i + 1; k += block_size) {
				for(int i1 = i; i1 < i + block_size; i1++){
					for(int j1 = j; j1 < j + block_size; j1++){
						int limit = min(k+block_size, i1+1);
						double temp = 0;
						for(int k1 = k; k1 < limit; k1++){
							temp += A[k1][i1] * B[j1][k1];
						}
						C[i1][j1] += temp;
					}
				}
			}
		}
	}
}

void optimized5(double** A, double** B, double** C) {
	int i, j, k;
	for (k = 0; k < N; k += block_size) {
		for (i = k; i < N; i += block_size) {
			for (j = 0; j < N; j += block_size) {
				for(int k1 = k; k1 < k + block_size; k1++){
					int limit = min(i+block_size, N);
					int start = max(k1, i);
					for(int i1 = start; i1 < limit; i1++){
						for(int j1 = j; j1 < j + block_size; j1++){
							C[i1][j1] += A[k1][i1] * B[j1][k1];
						}
					}
				}
			}
		}
	}
}

void optimized6(double** A, double** B, double** C) {
	int i, j, k;
	int unrolling_factor = unrolling_factor1;
	for (i = 0; i < N; i += block_size) {
		for (j = 0; j < N; j += block_size) {
			for (k = 0; k < i + 1; k += block_size) {
				for(int i1 = i; i1 < i + block_size; i1++){
					for(int j1 = j; j1 < j + block_size; j1++){
						int limit = min(k+block_size, i1+1);
						double temp = 0;
						for(int k1 = k; k1 + unrolling_factor -1 < limit; k1 += unrolling_factor){
							temp += A[k1][i1] * B[j1][k1];
							temp += A[k1+1][i1] * B[j1][k1+1];
						}
						for(int k1 = k + unrolling_factor * ((limit -k)/unrolling_factor); k1 < limit; k1++){
							temp += A[k1][i1] * B[j1][k1];
						}
						C[i1][j1] += temp;
					}
				}
			}
		}
	}
}

void optimized7(double** A, double** B, double** C) {
	int i, j, k;
	int unrolling_factor = unrolling_factor2;
	for (i = 0; i < N; i += block_size) {
		for (j = 0; j < N; j += block_size) {
			for (k = 0; k < i + 1; k += block_size) {
				for(int i1 = i; i1 < i + block_size; i1++){
					for(int j1 = j; j1 < j + block_size; j1++){
						int limit = min(k+block_size, i1+1);
						double temp = 0;
						for(int k1 = k; k1 + unrolling_factor -1 < limit; k1 += unrolling_factor){
							temp += A[k1][i1] * B[j1][k1];
							temp += A[k1+1][i1] * B[j1][k1+1];
							temp += A[k1+2][i1] * B[j1][k1+2];
							temp += A[k1+3][i1] * B[j1][k1+3];
						}
						for(int k1 = k + unrolling_factor * ((limit -k)/unrolling_factor); k1 < limit; k1++){
							temp += A[k1][i1] * B[j1][k1];
						}
						C[i1][j1] += temp;
					}
				}
			}
		}
	}
}

void optimized8(double** A, double** B, double** C) {
	int i, j, k;
	int unrolling_factor = unrolling_factor3;
	for (i = 0; i < N; i += block_size) {
		for (j = 0; j < N; j += block_size) {
			for (k = 0; k < i + 1; k += block_size) {
				for(int i1 = i; i1 < i + block_size; i1++){
					for(int j1 = j; j1 < j + block_size; j1++){
						int limit = min(k+block_size, i1+1);
						double temp = 0;
						for(int k1 = k; k1 + unrolling_factor -1 < limit; k1 += unrolling_factor){
							temp += A[k1][i1] * B[j1][k1];
							temp += A[k1+1][i1] * B[j1][k1+1];
							temp += A[k1+2][i1] * B[j1][k1+2];
							temp += A[k1+3][i1] * B[j1][k1+3];
							temp += A[k1+4][i1] * B[j1][k1+4];
							temp += A[k1+5][i1] * B[j1][k1+5];
							temp += A[k1+6][i1] * B[j1][k1+6];
							temp += A[k1+7][i1] * B[j1][k1+7];
						}
						for(int k1 = k + unrolling_factor * ((limit -k)/unrolling_factor); k1 < limit; k1++){
							temp += A[k1][i1] * B[j1][k1];
						}
						C[i1][j1] += temp;
					}
				}
			}
		}
	}
}

void optimized9(double** A, double** B, double** C) {
	int i, j, k;
	int unrolling_factor = unrolling_factor4;
	for (i = 0; i < N; i += block_size) {
		for (j = 0; j < N; j += block_size) {
			for (k = 0; k < i + 1; k += block_size) {
				for(int i1 = i; i1 < i + block_size; i1++){
					for(int j1 = j; j1 < j + block_size; j1++){
						int limit = min(k+block_size, i1+1);
						double temp = 0;
						for(int k1 = k; k1 + unrolling_factor -1 < limit; k1 += unrolling_factor){
							temp += A[k1][i1] * B[j1][k1];
							temp += A[k1+1][i1] * B[j1][k1+1];
							temp += A[k1+2][i1] * B[j1][k1+2];
							temp += A[k1+3][i1] * B[j1][k1+3];
							temp += A[k1+4][i1] * B[j1][k1+4];
							temp += A[k1+5][i1] * B[j1][k1+5];
							temp += A[k1+6][i1] * B[j1][k1+6];
							temp += A[k1+7][i1] * B[j1][k1+7];
							temp += A[k1+8][i1] * B[j1][k1+8];
							temp += A[k1+9][i1] * B[j1][k1+9];
							temp += A[k1+10][i1] * B[j1][k1+10];
							temp += A[k1+11][i1] * B[j1][k1+11];
							temp += A[k1+12][i1] * B[j1][k1+12];
							temp += A[k1+13][i1] * B[j1][k1+13];
							temp += A[k1+14][i1] * B[j1][k1+14];
							temp += A[k1+15][i1] * B[j1][k1+15];
						}
						for(int k1 = k + unrolling_factor * ((limit -k)/unrolling_factor); k1 < limit; k1++){
							temp += A[k1][i1] * B[j1][k1];
						}
						C[i1][j1] += temp;
					}
				}
			}
		}
	}
}

void optimized10(double** A, double** B, double** C) {
	int i, j, k;
	int unrolling_factor = unrolling_factor1;
	for (k = 0; k < N; k += block_size) {
		for (i = k; i < N; i += block_size) {
			for (j = 0; j < N; j += block_size) {
				for(int k1 = k; k1 < k + block_size; k1++){
					int limit = min(i+block_size, N);
					int start = max(k1, i);
					for(int i1 = start; i1 < limit; i1++){
						for(int j1 = j; j1 + unrolling_factor - 1 < j + block_size; j1 += unrolling_factor){
							C[i1][j1] += A[k1][i1] * B[j1][k1];
							C[i1][j1+1] += A[k1][i1] * B[j1+1][k1];
						}
						for(int j1 = j + unrolling_factor*(block_size/unrolling_factor); j1 < j + block_size; j1++){
							C[i1][j1] += A[k1][i1] * B[j1][k1];
						}
					}
				}
			}
		}
	}
}

void optimized11(double** A, double** B, double** C) {
	int i, j, k;
	int unrolling_factor = unrolling_factor2;
	for (k = 0; k < N; k += block_size) {
		for (i = k; i < N; i += block_size) {
			for (j = 0; j < N; j += block_size) {
				for(int k1 = k; k1 < k + block_size; k1++){
					int limit = min(i+block_size, N);
					int start = max(k1, i);
					for(int i1 = start; i1 < limit; i1++){
						for(int j1 = j; j1 + unrolling_factor - 1 < j + block_size; j1 += unrolling_factor){
							C[i1][j1] += A[k1][i1] * B[j1][k1];
							C[i1][j1+1] += A[k1][i1] * B[j1+1][k1];
							C[i1][j1+2] += A[k1][i1] * B[j1+2][k1];
							C[i1][j1+3] += A[k1][i1] * B[j1+3][k1];
						}
						for(int j1 = j + unrolling_factor*(block_size/unrolling_factor); j1 < j + block_size; j1++){
							C[i1][j1] += A[k1][i1] * B[j1][k1];
						}
					}
				}
			}
		}
	}
}

void optimized12(double** A, double** B, double** C) {
	int i, j, k;
	int unrolling_factor = unrolling_factor3;
	for (k = 0; k < N; k += block_size) {
		for (i = k; i < N; i += block_size) {
			for (j = 0; j < N; j += block_size) {
				for(int k1 = k; k1 < k + block_size; k1++){
					int limit = min(i+block_size, N);
					int start = max(k1, i);
					for(int i1 = start; i1 < limit; i1++){
						for(int j1 = j; j1 + unrolling_factor - 1 < j + block_size; j1 += unrolling_factor){
							C[i1][j1] += A[k1][i1] * B[j1][k1];
							C[i1][j1+1] += A[k1][i1] * B[j1+1][k1];
							C[i1][j1+2] += A[k1][i1] * B[j1+2][k1];
							C[i1][j1+3] += A[k1][i1] * B[j1+3][k1];
							C[i1][j1+4] += A[k1][i1] * B[j1+4][k1];
							C[i1][j1+5] += A[k1][i1] * B[j1+5][k1];
							C[i1][j1+6] += A[k1][i1] * B[j1+6][k1];
							C[i1][j1+7] += A[k1][i1] * B[j1+7][k1];
						}
						for(int j1 = j + unrolling_factor*(block_size/unrolling_factor); j1 < j + block_size; j1++){
							C[i1][j1] += A[k1][i1] * B[j1][k1];
						}
					}
				}
			}
		}
	}
}

void optimized13(double** A, double** B, double** C) {
	int i, j, k;
	int unrolling_factor = unrolling_factor4;
	for (k = 0; k < N; k += block_size) {
		for (i = k; i < N; i += block_size) {
			for (j = 0; j < N; j += block_size) {
				for(int k1 = k; k1 < k + block_size; k1++){
					int limit = min(i+block_size, N);
					int start = max(k1, i);
					for(int i1 = start; i1 < limit; i1++){
						for(int j1 = j; j1 + unrolling_factor - 1 < j + block_size; j1 += unrolling_factor){
							C[i1][j1] += A[k1][i1] * B[j1][k1];
							C[i1][j1+1] += A[k1][i1] * B[j1+1][k1];
							C[i1][j1+2] += A[k1][i1] * B[j1+2][k1];
							C[i1][j1+3] += A[k1][i1] * B[j1+3][k1];
							C[i1][j1+4] += A[k1][i1] * B[j1+4][k1];
							C[i1][j1+5] += A[k1][i1] * B[j1+5][k1];
							C[i1][j1+6] += A[k1][i1] * B[j1+6][k1];
							C[i1][j1+7] += A[k1][i1] * B[j1+7][k1];
							C[i1][j1+8] += A[k1][i1] * B[j1+8][k1];
							C[i1][j1+9] += A[k1][i1] * B[j1+9][k1];
							C[i1][j1+10] += A[k1][i1] * B[j1+10][k1];
							C[i1][j1+11] += A[k1][i1] * B[j1+11][k1];
							C[i1][j1+12] += A[k1][i1] * B[j1+12][k1];
							C[i1][j1+13] += A[k1][i1] * B[j1+13][k1];
							C[i1][j1+14] += A[k1][i1] * B[j1+14][k1];
							C[i1][j1+15] += A[k1][i1] * B[j1+15][k1];
						}
						for(int j1 = j + unrolling_factor*(block_size/unrolling_factor); j1 < j + block_size; j1++){
							C[i1][j1] += A[k1][i1] * B[j1][k1];
						}
					}
				}
			}
		}
	}
}

void intrinsics(double** A_intrinsics, double** B_intrinsics, double** C_intrinsics) {
	int i, j, k;
	for (i = 0; i < N; i += block_size) {
		for (j = 0; j < N; j += block_size) {
			for (k = 0; k < i + 1; k += block_size) {
				for(int i1 = i; i1 < i + block_size; i1++){
					for(int j1 = j; j1 < j + block_size; j1++){
						int limit = min(k+block_size, i1+1);
						double temp = 0;
						__m256d r1, r2, r3, r4;
						r3 = _mm256_set1_pd((double)0.0);
						for(int k1 = k; k1 + 3 < limit; k1 += 4){
							r1 = _mm256_set_pd(A_intrinsics[k1+3][i1], A_intrinsics[k1+2][i1],
								A_intrinsics[k1+1][i1], A_intrinsics[k1][i1]);
							r2 = _mm256_load_pd(&B_intrinsics[j1][k1]);
							r3 = _mm256_add_pd(r3, _mm256_mul_pd(r1, r2));
						}
						for(int k1 = k + 4 * ((limit -k)/4); k1 < limit; k1++){
							temp += A_intrinsics[k1][i1] * B_intrinsics[j1][k1];
						}
						r4 = _mm256_hadd_pd(r3, r3);
						temp += ((double*)&r4)[0] + ((double*)&r4)[2];
						C_intrinsics[i1][j1] += temp;
					}
				}
			}
		}
	}
}

int main(int argc, char* argv[]) {
	if (argc == 2) {
		block_size = atoi(argv[1]);
	}
	else {
		block_size = 4;
		cout << "Using default block size = 4\n";
	}

	double clkbegin, clkend;
	double t;

	int i, j, it;
	cout.setf(ios::fixed, ios::floatfield);
	cout.precision(5);

	double **A, **B, **C_ref, **C_opt;
	A = new double*[N];
	B = new double*[N];
	C_ref = new double*[N];
	C_opt = new double*[N];
	for (i = 0; i < N; i++) {
		A[i] = new double[N];
		B[i] = new double[N];
		C_ref[i] = new double[N];
		C_opt[i] = new double[N];
	}

	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			A[i][j] = i + j + 1;
			B[i][j] = (i + 1) * (j + 1);
			C_ref[i][j] = 0.0;
			C_opt[i][j] = 0.0;
		}
	}

	clkbegin = rtclock();
	for (it = 0; it < Niter; it++)
		reference(A, B, C_ref);
	clkend = rtclock();
	t = clkend - clkbegin;
	cout << "Reference Version: Matrix Size = " << N << ", " << 2.0 * 1e-9 * N * N * Niter / t
		<< " GFLOPS; Time = " << t / Niter << " sec\n";

	// clkbegin = rtclock();
	// for (it = 0; it < Niter; it++)
	// 	optimized1(A, B, C_opt);
	// clkend = rtclock();
	// t = clkend - clkbegin;
	// cout << "Optimized Version 1: Matrix Size = " << N << ", Time = " << t / Niter << " sec\n";
	// check_result(C_ref, C_opt);
	// // Reset
	// reset(C_opt);

	// clkbegin = rtclock();
	// for (it = 0; it < Niter; it++)
	// 	optimized2(A, B, C_opt);
	// clkend = rtclock();
	// t = clkend - clkbegin;
	// cout << "Optimized Version 2: Matrix Size = " << N << ", Time = " << t / Niter << " sec\n";
	// check_result(C_ref, C_opt);
	// // Reset
	// reset(C_opt);

	// clkbegin = rtclock();
	// for (it = 0; it < Niter; it++)
	// 	optimized3(A, B, C_opt);
	// clkend = rtclock();
	// t = clkend - clkbegin;
	// cout << "Optimized Version 3: Matrix Size = " << N << ", Time = " << t / Niter << " sec\n";
	// check_result(C_ref, C_opt);
	// // Reset
	// reset(C_opt);

	clkbegin = rtclock();
	for (it = 0; it < Niter; it++)
		optimized4(A, B, C_opt);
	clkend = rtclock();
	t = clkend - clkbegin;
	cout << "Optimized Version 4: Matrix Size = " << N << ", Time = " << t / Niter << " sec\n";
	check_result(C_ref, C_opt);
	// Reset
	reset(C_opt);

	// clkbegin = rtclock();
	// for (it = 0; it < Niter; it++)
	// 	optimized5(A, B, C_opt);
	// clkend = rtclock();
	// t = clkend - clkbegin;
	// cout << "Optimized Version 5: Matrix Size = " << N << ", Time = " << t / Niter << " sec\n";
	// check_result(C_ref, C_opt);
	// // Reset
	// reset(C_opt);

	// clkbegin = rtclock();
	// for (it = 0; it < Niter; it++)
	// 	optimized6(A, B, C_opt);
	// clkend = rtclock();
	// t = clkend - clkbegin;
	// cout << "Optimized Version 6: Matrix Size = " << N << ", Time = " << t / Niter << " sec\n";
	// check_result(C_ref, C_opt);
	// // Reset
	// reset(C_opt);

	// clkbegin = rtclock();
	// for (it = 0; it < Niter; it++)
	// 	optimized7(A, B, C_opt);
	// clkend = rtclock();
	// t = clkend - clkbegin;
	// cout << "Optimized Version 7: Matrix Size = " << N << ", Time = " << t / Niter << " sec\n";
	// check_result(C_ref, C_opt);
	// // Reset
	// reset(C_opt);

	// clkbegin = rtclock();
	// for (it = 0; it < Niter; it++)
	// 	optimized8(A, B, C_opt);
	// clkend = rtclock();
	// t = clkend - clkbegin;
	// cout << "Optimized Version 8: Matrix Size = " << N << ", Time = " << t / Niter << " sec\n";
	// check_result(C_ref, C_opt);
	// // Reset
	// reset(C_opt);

	// clkbegin = rtclock();
	// for (it = 0; it < Niter; it++)
	// 	optimized9(A, B, C_opt);
	// clkend = rtclock();
	// t = clkend - clkbegin;
	// cout << "Optimized Version 9: Matrix Size = " << N << ", Time = " << t / Niter << " sec\n";
	// check_result(C_ref, C_opt);
	// // Reset
	// reset(C_opt);

	// clkbegin = rtclock();
	// for (it = 0; it < Niter; it++)
	// 	optimized10(A, B, C_opt);
	// clkend = rtclock();
	// t = clkend - clkbegin;
	// cout << "Optimized Version 10: Matrix Size = " << N << ", Time = " << t / Niter << " sec\n";
	// check_result(C_ref, C_opt);
	// // Reset
	// reset(C_opt);

	// clkbegin = rtclock();
	// for (it = 0; it < Niter; it++)
	// 	optimized11(A, B, C_opt);
	// clkend = rtclock();
	// t = clkend - clkbegin;
	// cout << "Optimized Version 11: Matrix Size = " << N << ", Time = " << t / Niter << " sec\n";
	// check_result(C_ref, C_opt);
	// // Reset
	// reset(C_opt);

	// clkbegin = rtclock();
	// for (it = 0; it < Niter; it++)
	// 	optimized12(A, B, C_opt);
	// clkend = rtclock();
	// t = clkend - clkbegin;
	// cout << "Optimized Version 12: Matrix Size = " << N << ", Time = " << t / Niter << " sec\n";
	// check_result(C_ref, C_opt);
	// // Reset
	// reset(C_opt);

	// clkbegin = rtclock();
	// for (it = 0; it < Niter; it++)
	// 	optimized13(A, B, C_opt);
	// clkend = rtclock();
	// t = clkend - clkbegin;
	// cout << "Optimized Version 13: Matrix Size = " << N << ", Time = " << t / Niter << " sec\n";
	// check_result(C_ref, C_opt);
	// // Reset
	// reset(C_opt);

	// intrinsics version
	double **A_intrinsics __attribute__((aligned(32)));
	double **B_intrinsics __attribute__((aligned(32)));
	double **C_intrinsics __attribute__((aligned(32)));
	A_intrinsics = static_cast<double**>(aligned_alloc(32, N * sizeof(double*)));
	B_intrinsics = static_cast<double**>(aligned_alloc(32, N * sizeof(double*)));
	C_intrinsics = static_cast<double**>(aligned_alloc(32, N * sizeof(double*)));
	for (i = 0; i < N; i++) {
		A_intrinsics[i] = static_cast<double*>(aligned_alloc(32, N * sizeof(double)));
		B_intrinsics[i] = static_cast<double*>(aligned_alloc(32, N * sizeof(double)));
		C_intrinsics[i] = static_cast<double*>(aligned_alloc(32, N * sizeof(double)));
	}

	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			A_intrinsics[i][j] = i + j + 1;
			B_intrinsics[i][j] = (i + 1) * (j + 1);
			C_intrinsics[i][j] = 0.0;
		}
	}

	clkbegin = rtclock();
	for (it = 0; it < Niter; it++)
		intrinsics(A_intrinsics, B_intrinsics, C_intrinsics);
	clkend = rtclock();
	t = clkend - clkbegin;
	cout << "Intrinsics Version: Matrix Size = " << N << ", Time = " << t / Niter << " sec\n";
	check_result(C_ref, C_intrinsics);


	cout << endl;

	return EXIT_SUCCESS;
}
