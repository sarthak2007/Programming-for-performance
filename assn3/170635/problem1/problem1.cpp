// Compile: g++ -O2 -o problem1 problem1.cpp -mavx
// Execute: ./problem1 <block_size>

#include <cmath>
#include <iostream>
#include <sys/time.h>
#include <unistd.h>
#include <immintrin.h>

using namespace std;

const int N = 1 << 13;
const int Niter = 10;
const double THRESHOLD = 0.000001;

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

void check_result(double* w_ref, double* w_opt) {
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

void reset(double* y_opt, double* z_opt){
	for (int i = 0; i < N; i++) {
		y_opt[i] = 1.0;
		z_opt[i] = 2.0;
	}
}

void reference(double** A, double* x, double* y_ref, double* z_ref) {
	int i, j;
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			y_ref[j] = y_ref[j] + A[i][j] * x[i];
			z_ref[j] = z_ref[j] + A[j][i] * x[i];
		}
	}
}

// TODO: INITIALLY IDENTICAL TO REFERENCE; MAKE YOUR CHANGES TO OPTIMIZE THIS CODE
// You can create multiple versions of the optimized() function to test your changes
void optimized1(double** A, double* x, double* y_opt, double* z_opt) {
	int i, j;
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			y_opt[j] = y_opt[j] + A[i][j] * x[i];
		}
	}
	for (j = 0; j < N; j++) {
		double temp = 0;
		for (i = 0; i < N; i++) {
			temp = temp + A[j][i] * x[i];
		}
		z_opt[j] += temp;
	}
}

void optimized2(double** A, double* x, double* y_opt, double* z_opt) {
	int i, j;
	for (i = 0; i < N; i += block_size) {
		for (j = 0; j < N; j += block_size) {
			for(int i1 = i; i1 < i + block_size; i1++){
				for(int j1 = j; j1 < j + block_size; j1++){
					y_opt[j1] = y_opt[j1] + A[i1][j1] * x[i1];
				}
			}
		}
	}
	for (j = 0; j < N; j += block_size) {
		for (i = 0; i < N; i += block_size) {
			for(int j1 = j; j1 < j + block_size; j1++){
				double temp = 0;
				for(int i1 = i; i1 < i + block_size; i1++){
					temp = temp + A[j1][i1] * x[i1];
				}
				z_opt[j1] += temp;
			}
		}
	}
}

void optimized3(double** A, double* x, double* y_opt, double* z_opt) {
	int i, j;
	int unrolling_factor = unrolling_factor1;
	for (i = 0; i < N; i += block_size) {
		for (j = 0; j < N; j += block_size) {
			for(int i1 = i; i1 < i + block_size; i1++){
				for(int j1 = j; j1 + unrolling_factor - 1 < j + block_size; j1 += unrolling_factor){
					y_opt[j1] += A[i1][j1] * x[i1];
					y_opt[j1+1] += A[i1][j1+1] * x[i1];
				}
				for(int j1 = j + unrolling_factor * (block_size/unrolling_factor); j1 < j + block_size; j1++){
					y_opt[j1] = y_opt[j1] + A[i1][j1] * x[i1];
				}
			}
		}
	}
	for (j = 0; j < N; j += block_size) {
		for (i = 0; i < N; i += block_size) {
			for(int j1 = j; j1 < j + block_size; j1++){
				double temp = 0;
				for(int i1 = i; i1 + unrolling_factor -1 < i + block_size; i1 += unrolling_factor){
					temp = temp + A[j1][i1] * x[i1];
					temp = temp + A[j1][i1+1] * x[i1+1];
				}
				for(int i1 = i + unrolling_factor * (block_size/unrolling_factor); i1 < i + block_size; i1++){
					temp = temp + A[j1][i1] * x[i1];
				}
				z_opt[j1] += temp;
			}
		}
	}
}

void optimized4(double** A, double* x, double* y_opt, double* z_opt) {
	int i, j;
	int unrolling_factor = unrolling_factor2;
	for (i = 0; i < N; i += block_size) {
		for (j = 0; j < N; j += block_size) {
			for(int i1 = i; i1 < i + block_size; i1++){
				for(int j1 = j; j1 + unrolling_factor - 1 < j + block_size; j1 += unrolling_factor){
					y_opt[j1] += A[i1][j1] * x[i1];
					y_opt[j1+1] += A[i1][j1+1] * x[i1];
					y_opt[j1+2] += A[i1][j1+2] * x[i1];
					y_opt[j1+3] += A[i1][j1+3] * x[i1];
				}
				for(int j1 = j + unrolling_factor * (block_size/unrolling_factor); j1 < j + block_size; j1++){
					y_opt[j1] = y_opt[j1] + A[i1][j1] * x[i1];
				}
			}
		}
	}
	for (j = 0; j < N; j += block_size) {
		for (i = 0; i < N; i += block_size) {
			for(int j1 = j; j1 < j + block_size; j1++){
				double temp = 0;
				for(int i1 = i; i1 + unrolling_factor -1 < i + block_size; i1 += unrolling_factor){
					temp = temp + A[j1][i1] * x[i1];
					temp = temp + A[j1][i1+1] * x[i1+1];
					temp = temp + A[j1][i1+2] * x[i1+2];
					temp = temp + A[j1][i1+3] * x[i1+3];
				}
				for(int i1 = i + unrolling_factor * (block_size/unrolling_factor); i1 < i + block_size; i1++){
					temp = temp + A[j1][i1] * x[i1];
				}
				z_opt[j1] += temp;
			}
		}
	}
}

void optimized5(double** A, double* x, double* y_opt, double* z_opt) {
	int i, j;
	int unrolling_factor = unrolling_factor3;
	for (i = 0; i < N; i += block_size) {
		for (j = 0; j < N; j += block_size) {
			for(int i1 = i; i1 < i + block_size; i1++){
				for(int j1 = j; j1 + unrolling_factor - 1 < j + block_size; j1 += unrolling_factor){
					y_opt[j1] += A[i1][j1] * x[i1];
					y_opt[j1+1] += A[i1][j1+1] * x[i1];
					y_opt[j1+2] += A[i1][j1+2] * x[i1];
					y_opt[j1+3] += A[i1][j1+3] * x[i1];
					y_opt[j1+4] += A[i1][j1+4] * x[i1];
					y_opt[j1+5] += A[i1][j1+5] * x[i1];
					y_opt[j1+6] += A[i1][j1+6] * x[i1];
					y_opt[j1+7] += A[i1][j1+7] * x[i1];
				}
				for(int j1 = j + unrolling_factor * (block_size/unrolling_factor); j1 < j + block_size; j1++){
					y_opt[j1] = y_opt[j1] + A[i1][j1] * x[i1];
				}
			}
		}
	}
	for (j = 0; j < N; j += block_size) {
		for (i = 0; i < N; i += block_size) {
			for(int j1 = j; j1 < j + block_size; j1++){
				double temp = 0;
				for(int i1 = i; i1 + unrolling_factor -1 < i + block_size; i1 += unrolling_factor){
					temp = temp + A[j1][i1] * x[i1];
					temp = temp + A[j1][i1+1] * x[i1+1];
					temp = temp + A[j1][i1+2] * x[i1+2];
					temp = temp + A[j1][i1+3] * x[i1+3];
					temp = temp + A[j1][i1+4] * x[i1+4];
					temp = temp + A[j1][i1+5] * x[i1+5];
					temp = temp + A[j1][i1+6] * x[i1+6];
					temp = temp + A[j1][i1+7] * x[i1+7];
				}
				for(int i1 = i + unrolling_factor * (block_size/unrolling_factor); i1 < i + block_size; i1++){
					temp = temp + A[j1][i1] * x[i1];
				}
				z_opt[j1] += temp;
			}
		}
	}
}

void optimized6(double** A, double* x, double* y_opt, double* z_opt) {
	int i, j;
	int unrolling_factor = unrolling_factor4;
	for (i = 0; i < N; i += block_size) {
		for (j = 0; j < N; j += block_size) {
			for(int i1 = i; i1 < i + block_size; i1++){
				for(int j1 = j; j1 + unrolling_factor - 1 < j + block_size; j1 += unrolling_factor){
					y_opt[j1] += A[i1][j1] * x[i1];
					y_opt[j1+1] += A[i1][j1+1] * x[i1];
					y_opt[j1+2] += A[i1][j1+2] * x[i1];
					y_opt[j1+3] += A[i1][j1+3] * x[i1];
					y_opt[j1+4] += A[i1][j1+4] * x[i1];
					y_opt[j1+5] += A[i1][j1+5] * x[i1];
					y_opt[j1+6] += A[i1][j1+6] * x[i1];
					y_opt[j1+7] += A[i1][j1+7] * x[i1];
					y_opt[j1+8] += A[i1][j1+8] * x[i1];
					y_opt[j1+9] += A[i1][j1+9] * x[i1];
					y_opt[j1+10] += A[i1][j1+10] * x[i1];
					y_opt[j1+11] += A[i1][j1+11] * x[i1];
					y_opt[j1+12] += A[i1][j1+12] * x[i1];
					y_opt[j1+13] += A[i1][j1+13] * x[i1];
					y_opt[j1+14] += A[i1][j1+14] * x[i1];
					y_opt[j1+15] += A[i1][j1+15] * x[i1];
				}
				for(int j1 = j + unrolling_factor * (block_size/unrolling_factor); j1 < j + block_size; j1++){
					y_opt[j1] = y_opt[j1] + A[i1][j1] * x[i1];
				}
			}
		}
	}
	for (j = 0; j < N; j += block_size) {
		for (i = 0; i < N; i += block_size) {
			for(int j1 = j; j1 < j + block_size; j1++){
				double temp = 0;
				for(int i1 = i; i1 + unrolling_factor -1 < i + block_size; i1 += unrolling_factor){
					temp = temp + A[j1][i1] * x[i1];
					temp = temp + A[j1][i1+1] * x[i1+1];
					temp = temp + A[j1][i1+2] * x[i1+2];
					temp = temp + A[j1][i1+3] * x[i1+3];
					temp = temp + A[j1][i1+4] * x[i1+4];
					temp = temp + A[j1][i1+5] * x[i1+5];
					temp = temp + A[j1][i1+6] * x[i1+6];
					temp = temp + A[j1][i1+7] * x[i1+7];
					temp = temp + A[j1][i1+8] * x[i1+8];
					temp = temp + A[j1][i1+9] * x[i1+9];
					temp = temp + A[j1][i1+10] * x[i1+10];
					temp = temp + A[j1][i1+11] * x[i1+11];
					temp = temp + A[j1][i1+12] * x[i1+12];
					temp = temp + A[j1][i1+13] * x[i1+13];
					temp = temp + A[j1][i1+14] * x[i1+14];
					temp = temp + A[j1][i1+15] * x[i1+15];
				}
				for(int i1 = i + unrolling_factor * (block_size/unrolling_factor); i1 < i + block_size; i1++){
					temp = temp + A[j1][i1] * x[i1];
				}
				z_opt[j1] += temp;
			}
		}
	}
}

void intrinsics(double** A_intrinsics, double* x_intrinsics, double* y_intrinsics, double* z_intrinsics) {
	int i, j;
	int unrolling_factor = 16;
	for (i = 0; i < N; i += block_size) {
		for (j = 0; j < N; j += block_size) {
			for(int i1 = i; i1 < i + block_size; i1++){
				__m256d r1, r2, r3, r4;
				r4 = _mm256_set1_pd(x_intrinsics[i1]);
				for(int j1 = j; j1 + unrolling_factor - 1 < j + block_size; j1 += unrolling_factor){
					r1 = _mm256_load_pd(&y_intrinsics[j1]);
					r2 = _mm256_load_pd(&A_intrinsics[i1][j1]);
					r3 = _mm256_add_pd(r1, _mm256_mul_pd(r2, r4));
					_mm256_store_pd(&y_intrinsics[j1], r3);

					r1 = _mm256_load_pd(&y_intrinsics[j1+4]);
					r2 = _mm256_load_pd(&A_intrinsics[i1][j1+4]);
					r3 = _mm256_add_pd(r1, _mm256_mul_pd(r2, r4));
					_mm256_store_pd(&y_intrinsics[j1+4], r3);

					r1 = _mm256_load_pd(&y_intrinsics[j1+8]);
					r2 = _mm256_load_pd(&A_intrinsics[i1][j1+8]);
					r3 = _mm256_add_pd(r1, _mm256_mul_pd(r2, r4));
					_mm256_store_pd(&y_intrinsics[j1+8], r3);

					r1 = _mm256_load_pd(&y_intrinsics[j1+12]);
					r2 = _mm256_load_pd(&A_intrinsics[i1][j1+12]);
					r3 = _mm256_add_pd(r1, _mm256_mul_pd(r2, r4));
					_mm256_store_pd(&y_intrinsics[j1+12], r3);
				}
				for(int j1 = j + unrolling_factor * (block_size/unrolling_factor); j1 < j + block_size; j1++){
					y_intrinsics[j1] = y_intrinsics[j1] + A_intrinsics[i1][j1] * x_intrinsics[i1];
				}
			}
		}
	}
	for (j = 0; j < N; j += block_size) {
		for (i = 0; i < N; i += block_size) {
			for(int j1 = j; j1 < j + block_size; j1++){
				double temp = 0;
				__m256d r1, r2, r3, r4;
				r3 = _mm256_set1_pd((double)0.0);
				for(int i1 = i; i1 + unrolling_factor -1 < i + block_size; i1 += unrolling_factor){
					r1 = _mm256_load_pd(&A_intrinsics[j1][i1]);
					r2 = _mm256_load_pd(&x_intrinsics[i1]);
					r3 = _mm256_add_pd(r3, _mm256_mul_pd(r1, r2));

					r1 = _mm256_load_pd(&A_intrinsics[j1][i1+4]);
					r2 = _mm256_load_pd(&x_intrinsics[i1+4]);
					r3 = _mm256_add_pd(r3, _mm256_mul_pd(r1, r2));

					r1 = _mm256_load_pd(&A_intrinsics[j1][i1+8]);
					r2 = _mm256_load_pd(&x_intrinsics[i1+8]);
					r3 = _mm256_add_pd(r3, _mm256_mul_pd(r1, r2));

					r1 = _mm256_load_pd(&A_intrinsics[j1][i1+12]);
					r2 = _mm256_load_pd(&x_intrinsics[i1+12]);
					r3 = _mm256_add_pd(r3, _mm256_mul_pd(r1, r2));
				}
				r4 = _mm256_hadd_pd(r3, r3);
				temp = ((double*)&r4)[0] + ((double*)&r4)[2];
				for(int i1 = i + unrolling_factor * (block_size/unrolling_factor); i1 < i + block_size; i1++){
					temp = temp + A_intrinsics[j1][i1] * x_intrinsics[i1];
				}
				z_intrinsics[j1] += temp;
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

	double** A;
	A = new double*[N];
	for (int i = 0; i < N; i++) {
		A[i] = new double[N];
	}

	double *x, *y_ref, *z_ref, *y_opt, *z_opt;
	x = new double[N];
	y_ref = new double[N];
	z_ref = new double[N];
	y_opt = new double[N];
	z_opt = new double[N];

	for (i = 0; i < N; i++) {
		x[i] = i;
		y_ref[i] = 1.0;
		y_opt[i] = 1.0;
		z_ref[i] = 2.0;
		z_opt[i] = 2.0;
		for (j = 0; j < N; j++) {
			A[i][j] = (i + 2.0 * j) / (2.0 * N);
		}
	}

	clkbegin = rtclock();
	for (it = 0; it < Niter; it++) {
		reference(A, x, y_ref, z_ref);
	}
	clkend = rtclock();
	t = clkend - clkbegin;
	cout << "Reference Version: Matrix Size = " << N << ", " << 4.0 * 1e-9 * N * N * Niter / t
		<< " GFLOPS; Time = " << t / Niter << " sec\n";

	clkbegin = rtclock();
	for (it = 0; it < Niter; it++) {
		optimized1(A, x, y_opt, z_opt);
	}
	clkend = rtclock();
	t = clkend - clkbegin;
	cout << "Optimized Version 1: Matrix Size = " << N << ", Time = " << t / Niter << " sec\n";
	check_result(y_ref, y_opt);
	check_result(z_ref, z_opt);
	// Reset
	reset(y_opt, z_opt);

	// clkbegin = rtclock();
	// for (it = 0; it < Niter; it++) {
	// 	optimized2(A, x, y_opt, z_opt);
	// }
	// clkend = rtclock();
	// t = clkend - clkbegin;
	// cout << "Optimized Version 2: Matrix Size = " << N << ", Time = " << t / Niter << " sec\n";
	// check_result(y_ref, y_opt);
	// check_result(z_ref, z_opt);
	// // Reset
	// reset(y_opt, z_opt);

	// clkbegin = rtclock();
	// for (it = 0; it < Niter; it++) {
	// 	optimized3(A, x, y_opt, z_opt);
	// }
	// clkend = rtclock();
	// t = clkend - clkbegin;
	// cout << "Optimized Version 3: Matrix Size = " << N << ", Time = " << t / Niter << " sec\n";
	// check_result(y_ref, y_opt);
	// check_result(z_ref, z_opt);
	// // Reset
	// reset(y_opt, z_opt);

	// clkbegin = rtclock();
	// for (it = 0; it < Niter; it++) {
	// 	optimized4(A, x, y_opt, z_opt);
	// }
	// clkend = rtclock();
	// t = clkend - clkbegin;
	// cout << "Optimized Version 4: Matrix Size = " << N << ", Time = " << t / Niter << " sec\n";
	// check_result(y_ref, y_opt);
	// check_result(z_ref, z_opt);
	// // Reset
	// reset(y_opt, z_opt);

	// clkbegin = rtclock();
	// for (it = 0; it < Niter; it++) {
	// 	optimized5(A, x, y_opt, z_opt);
	// }
	// clkend = rtclock();
	// t = clkend - clkbegin;
	// cout << "Optimized Version 5: Matrix Size = " << N << ", Time = " << t / Niter << " sec\n";
	// check_result(y_ref, y_opt);
	// check_result(z_ref, z_opt);
	// // Reset
	// reset(y_opt, z_opt);

	clkbegin = rtclock();
	for (it = 0; it < Niter; it++) {
		optimized6(A, x, y_opt, z_opt);
	}
	clkend = rtclock();
	t = clkend - clkbegin;
	cout << "Optimized Version 6: Matrix Size = " << N << ", Time = " << t / Niter << " sec\n";
	check_result(y_ref, y_opt);
	check_result(z_ref, z_opt);
	// Reset
	reset(y_opt, z_opt);

	// intrinsics version
	if(block_size >= 4){
		double** A_intrinsics __attribute__((aligned(32)));
		A_intrinsics = static_cast<double**>(aligned_alloc(32, N * sizeof(double*)));
		for (int i = 0; i < N; i++) {
			A_intrinsics[i] = static_cast<double*>(aligned_alloc(32, N * sizeof(double)));
		}
		double *x_intrinsics __attribute__((aligned(32)));
		double *y_intrinsics __attribute__((aligned(32)));
		double *z_intrinsics __attribute__((aligned(32)));
		x_intrinsics = static_cast<double*>(aligned_alloc(32, N * sizeof(double)));
		y_intrinsics = static_cast<double*>(aligned_alloc(32, N * sizeof(double)));
		z_intrinsics = static_cast<double*>(aligned_alloc(32, N * sizeof(double)));

		for (int i = 0; i < N; i++) {
			x_intrinsics[i] = i;
			y_intrinsics[i] = 1.0;
			z_intrinsics[i] = 2.0;
			for (j = 0; j < N; j++) {
				A_intrinsics[i][j] = (i + 2.0 * j) / (2.0 * N);
			}
		}

		clkbegin = rtclock();
		for (it = 0; it < Niter; it++) {
			intrinsics(A_intrinsics, x_intrinsics, y_intrinsics, z_intrinsics);
		}
		clkend = rtclock();
		t = clkend - clkbegin;
		cout << "Intrinsics Version: Matrix Size = " << N << ", Time = " << t / Niter << " sec\n";
		check_result(y_ref, y_intrinsics);
		check_result(z_ref, z_intrinsics);		
	}

	cout << '\n';

	return EXIT_SUCCESS;
}
