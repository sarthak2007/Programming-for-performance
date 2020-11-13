// Compile: g++ -std=c++11 -fopenmp fibonacci.cpp -o fibonacci -ltbb
// Execute: ./fibonacci <num_threads> <threshold>

#include <cassert>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include </usr/include/tbb/task.h>

#define N 50

using std::cout;
using std::endl;
using std::chrono::duration_cast;
using HR = std::chrono::high_resolution_clock;
using HRTimer = HR::time_point;
using std::chrono::microseconds;
using namespace tbb;

int num_threads; //taken from command line
int threshold; //taken from command line

// Serial Fibonacci
long ser_fib(int n) {
  if (n == 0 || n == 1) {
    return (n);
  }
  return ser_fib(n - 1) + ser_fib(n - 2);
}

// auxilary function for version 1 of OpenMP
long omp_fib_v1_aux(int n){
  if(n <= 1){
    return n;
  }
  long val1, val2;

  #pragma omp task shared(val1, n)
  val1 = omp_fib_v1_aux(n-1);

  #pragma omp task shared(val2, n)
  val2 = omp_fib_v1_aux(n-2);

  #pragma omp taskwait

  return val1 + val2;
}

// version 1 of OpenMP
long omp_fib_v1(int n) {
  long val;
  #pragma omp parallel num_threads(num_threads)
  {
    #pragma omp single nowait
    {
      val = omp_fib_v1_aux(n);
    }
  }
  return val;
}

// auxilary function for version 2 of OpenMP
long omp_fib_v2_aux(int n){
  if(n <= 1)
    return n;
  if(n <= threshold){
    return omp_fib_v2_aux(n-1) + omp_fib_v2_aux(n-2);
  }
  long val1, val2;

  #pragma omp task shared(val1, n)
  val1 = omp_fib_v2_aux(n-1);

  #pragma omp task shared(val2, n)
  val2 = omp_fib_v2_aux(n-2);

  #pragma omp taskwait

  return val1 + val2;
}

// version 2 of OpenMP
long omp_fib_v2(int n) {
  long val;
  #pragma omp parallel num_threads(num_threads)
  {
    #pragma omp single nowait
    {
      val = omp_fib_v2_aux(n);
    }
  }
  return val;
}

class fib_blocking: public task{
  public:
    long n, *val;
    fib_blocking(long n, long *val){
      this->n = n;
      this->val = val;
    }

    task* execute(){
      if(n <= threshold)
        *val = ser_fib(n);
      else{
        long val1, val2;

        fib_blocking& task1 = *new(allocate_child()) fib_blocking(n-1, &val1);
        fib_blocking& task2 = *new(allocate_child()) fib_blocking(n-2, &val2);

        set_ref_count(3);
        spawn(task1);
        spawn_and_wait_for_all(task2);
        *val = val1 + val2;
      }
      return NULL;
    }
};

long tbb_fib_blocking(int n) {
  long val;
  fib_blocking& task = *new(task::allocate_root()) fib_blocking(n, &val);
  task::spawn_root_and_wait(task);
  return val;   
}

class fib_continuation: public task{
  public:
    long val1, val2, *val;

    fib_continuation(long *val){
      this->val = val;
    }

    task* execute(){
      *val = val1 + val2;
      return NULL;
    }
};

class fib_cps: public task{
  public:
    long n, *val;

    fib_cps(long n, long *val){
      this->n = n;
      this->val = val;
    }

    task* execute(){
      if(n <= threshold)
        *val = ser_fib(n);
      else{

        fib_continuation& task_continue = *new(allocate_continuation()) fib_continuation(val);
        fib_cps& task1 = *new(task_continue.allocate_child()) fib_cps(n-1, &task_continue.val1);
        fib_cps& task2 = *new(task_continue.allocate_child()) fib_cps(n-2, &task_continue.val2);

        task_continue.set_ref_count(2);
        spawn(task1);

        return &task2;
      }
      return NULL;
    }
};

long tbb_fib_cps(int n) {
  long val;
  fib_cps& task = *new(task::allocate_root()) fib_cps(n, &val);
  task::spawn_root_and_wait(task);
  return val;   
}

int main(int argc, char** argv) {
  if(argc == 3){
		num_threads = atoi(argv[1]);
		threshold = atoi(argv[2]);
	}
  else if(argc == 2){
    num_threads = atoi(argv[1]);
    threshold = 10;
		cout << "Using default threshold = 10\n";
  }
	else{
		num_threads = 1;
    threshold = 10;
		cout << "Using default number of threads = 1 and threshold = 10\n";
	}

  cout << std::fixed << std::setprecision(5);

  HRTimer start = HR::now();
  long s_fib = ser_fib(N);
  HRTimer end = HR::now();
  auto duration = duration_cast<microseconds>(end - start).count();
  cout << "Serial time: " << duration << " microseconds" << endl;

  start = HR::now();
  long omp_v1 = omp_fib_v1(N);
  end = HR::now();
  assert(s_fib == omp_v1);
  duration = duration_cast<microseconds>(end - start).count();
  cout << "OMP v1 time: " << duration << " microseconds" << endl;

  start = HR::now();
  long omp_v2 = omp_fib_v2(N);
  end = HR::now();
  assert(s_fib == omp_v2);
  duration = duration_cast<microseconds>(end - start).count();
  cout << "OMP v2 time: " << duration << " microseconds" << endl;

  start = HR::now();
  long blocking_fib = tbb_fib_blocking(N);
  end = HR::now();
  assert(s_fib == blocking_fib);
  duration = duration_cast<microseconds>(end - start).count();
  cout << "TBB (blocking) time: " << duration << " microseconds" << endl;

  start = HR::now();
  long cps_fib = tbb_fib_cps(N);
  end = HR::now();
  assert(s_fib == cps_fib);
  duration = duration_cast<microseconds>(end - start).count();
  cout << "TBB (cps) time: " << duration << " microseconds" << endl;

  return EXIT_SUCCESS;
}
