// Compile: g++ -std=c++11 -fopenmp pi.cpp -o pi -ltbb
// Execute: ./pi <num_threads>

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <omp.h>
#include </usr/include/tbb/parallel_reduce.h>
#include </usr/include/tbb/blocked_range.h>

using std::cout;
using std::endl;
using std::chrono::duration_cast;
using HR = std::chrono::high_resolution_clock;
using HRTimer = HR::time_point;
using std::chrono::microseconds;
using namespace tbb;

const int NUM_INTERVALS = std::numeric_limits<int>::max();
int num_threads; //taken from command line

double serial_pi() {
  double dx = 1.0 / NUM_INTERVALS;
  double sum = 0.0;
  for (int i = 0; i < NUM_INTERVALS; ++i) {
    double x = (i + 0.5) * dx;
    double h = std::sqrt(1 - x * x);
    sum += h * dx;
  }
  double pi = 4 * sum;
  return pi;
}

double omp_pi() {
  double dx = 1.0 / NUM_INTERVALS;
  double sum = 0.0;

  omp_set_num_threads(num_threads);
  #pragma omp parallel for default(none) reduction(+:sum) shared(dx)
  for (int i = 0; i < NUM_INTERVALS; ++i) {
    double x = (i + 0.5) * dx;
    double h = std::sqrt(1 - x * x);
    sum += h * dx;
  }
  double pi = 4 * sum;
  return pi;
}

class calc_pi{
  public:
    double val;

    calc_pi(){
      this->val = 0;
    }

    calc_pi(calc_pi& other, split){
      this->val = 0;
    }

    void operator()(const blocked_range<size_t>& r){
      double temp = val;
      double dx = 1.0 / NUM_INTERVALS;
      for(size_t i = r.begin(); i != r.end(); i++){
        double x = (i + 0.5) * dx;
        double h = std::sqrt(1 - x * x);
        temp += h * dx;
      } 
      val = temp;      
    }

    void join(const calc_pi& other){
      val += other.val;
    }
};

double tbb_pi() {
  calc_pi obj;
  parallel_reduce(blocked_range<size_t>(0, NUM_INTERVALS), obj);
  return 4 * obj.val;
}

int main(int argc, char** argv) {
  if(argc == 2){
		num_threads = atoi(argv[1]);
	}
	else{
		num_threads = 1;
		cout << "Using default number of threads = 1\n";
	}

  cout << std::fixed << std::setprecision(5);

  HRTimer start = HR::now();
  double ser_pi = serial_pi();
  HRTimer end = HR::now();
  auto duration = duration_cast<microseconds>(end - start).count();
  cout << "Serial pi: " << ser_pi << " Serial time: " << duration << " microseconds" << endl;

  start = HR::now();
  double o_pi = omp_pi();
  end = HR::now();
  duration = duration_cast<microseconds>(end - start).count();
  cout << "Parallel pi (OMP): " << o_pi << " Parallel time: " << duration << " microseconds"
       << endl;

  start = HR::now();
  double t_pi = tbb_pi();
  end = HR::now();
  duration = duration_cast<microseconds>(end - start).count();
  cout << "Parallel pi (TBB): " << t_pi << " Parallel time: " << duration << " microseconds"
       << endl;

  return EXIT_SUCCESS;
}