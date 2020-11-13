// Compile: g++ -std=c++11 -fopenmp quicksort.cpp -o quicksort
// Execute: ./quicksort <num_threads> <threshold>

#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <omp.h>

using std::cout;
using std::endl;
using std::chrono::duration_cast;
using HR = std::chrono::high_resolution_clock;
using HRTimer = HR::time_point;
using std::chrono::microseconds;

// Make sure to test with other sizes
#define N (1 << 21)

int num_threads; //taken from command line
int threshold; //taken from command line

void swap(int* x, int* y) {
  int tmp = *x;
  *x = *y;
  *y = tmp;
}

int partition(int* arr, int low, int high) {
  int pivot, last;
  pivot = arr[low];
  swap(arr + low, arr + high);
  last = low;
  for (int i = low; i < high; i++) {
    if (arr[i] <= pivot) {
      swap(arr + last, arr + i);
      last += 1;
    }
  }
  swap(arr + last, arr + high);
  return last;
}

void serial_quicksort(int* arr, int start, int end) {
  int part;
  if (start < end) {
    part = partition(arr, start, end);

    serial_quicksort(arr, start, part - 1);
    serial_quicksort(arr, part + 1, end);
  }
}

int par_partition(int* arr, int low, int high) {
  int pivot = arr[low];
  int *temp = new int[high-low+1];
  int l = 0, r = high-low;

  #pragma omp parallel for
  for (int i = low+1; i <= high; i++) {
    if (arr[i] <= pivot) {
      #pragma omp critical
      {
        temp[l++] = arr[i];
      }
    }
    else{
      #pragma omp critical
      {
        temp[r--] = arr[i];
      }
    }
  }
  temp[l] = pivot;

  #pragma omp parallel for
  for(int i = low; i <= high; i++){
    arr[i] = temp[i-low];
  }
  delete[] temp;
  return l+low;
}

void par_quicksort_aux(int* arr, int start, int end) {
  int part;
  if(start < end){
    if(end - start <= threshold){
      serial_quicksort(arr, start, end);
      return;
    } 

    part = partition(arr, start, end);

    #pragma omp task
    par_quicksort_aux(arr, start, part - 1);

    #pragma omp task
    par_quicksort_aux(arr, part + 1, end);

    #pragma omp taskwait
  }
}

void par_quicksort(int* arr, int start, int end){
  omp_set_num_threads(num_threads);
  #pragma omp parallel
  {
    #pragma omp single nowait
    {
      par_quicksort_aux(arr, start, end);
    }
  }
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
  
  int* ser_arr = nullptr;
  int* par_arr = nullptr;
  ser_arr = new int[N];
  par_arr = new int[N];
  for (int i = 0; i < N; i++) {
    ser_arr[i] = rand() % 1000;
    par_arr[i] = ser_arr[i];
  }

  cout << "Unsorted array: " << endl;
  for (int i = 0; i < N; i++) {
    cout << ser_arr[i] << "\t";
  }
  cout << endl << endl;

  HRTimer start = HR::now();
  serial_quicksort(ser_arr, 0, N - 1);
  HRTimer end = HR::now();
  auto duration = duration_cast<microseconds>(end - start).count();
  cout << "Serial quicksort time: " << duration << " us" << endl;

  cout << "Sorted array: " << endl;
  for (int i = 0; i < N; i++) {
    cout << ser_arr[i] << "\t";
  }
  cout << endl << endl;

  start = HR::now();
  par_quicksort(par_arr, 0, N - 1);
  end = HR::now();
  duration = duration_cast<microseconds>(end - start).count();
  cout << "OpenMP quicksort time: " << duration << " us" << endl;

  for (int i = 0; i < N; i++) {
    assert(ser_arr[i] == par_arr[i]);
  }

  return EXIT_SUCCESS;
}
