// Compile: g++ -std=c++11 find-max.cpp -o find-max -ltbb
// Execute: ./find-max

#include <cassert>
#include <chrono>
#include <iostream>
#include </usr/include/tbb/parallel_reduce.h>
#include </usr/include/tbb/blocked_range.h>

using std::cout;
using std::endl;
using std::chrono::duration_cast;
using HR = std::chrono::high_resolution_clock;
using HRTimer = HR::time_point;
using std::chrono::microseconds;
using namespace tbb;

#define N (1 << 26)

uint32_t serial_find_max(const uint32_t* a) {
  uint32_t value_of_max = 0;
  uint32_t index_of_max = 0;
  for (uint32_t i = 0; i < N; i++) {
    uint32_t value = a[i];
    if (value > value_of_max) {
      value_of_max = value;
      index_of_max = i;
    }
  }
  return index_of_max;
}

class find_max{
    const uint32_t* a;

  public:
    uint32_t value_of_max, index_of_max;

    find_max(const uint32_t* a){
      this->a = a;
      this->value_of_max = this->index_of_max = 0;
    }

    find_max(find_max& other, split){
      a = other.a;
      value_of_max = index_of_max = 0;
    }

    void operator()(const blocked_range<size_t>& r){
      uint32_t temp_val = value_of_max;
      uint32_t temp_index = index_of_max;
      for(size_t i = r.begin(); i != r.end(); i++){
        uint32_t value = a[i];
        if (value > temp_val or (value == temp_val and i < temp_index)) {
          temp_val = value;
          temp_index = i;
        }
      } 
      value_of_max = temp_val;
      index_of_max = temp_index;      
    }

    void join(const find_max& other){
      if(other.value_of_max > value_of_max or (other.value_of_max == value_of_max and other.index_of_max < index_of_max)){
        value_of_max = other.value_of_max;
        index_of_max = other.index_of_max;
      }
    }
};

uint32_t tbb_find_max(const uint32_t* a) {
  find_max obj(a);
  parallel_reduce(blocked_range<size_t>(0, N), obj);
  return obj.index_of_max;
}

int main() {
  uint32_t* a = new uint32_t[N];
  for (uint32_t i = 0; i < N; i++) {
    a[i] = i;
  }

  HRTimer start = HR::now();
  uint64_t s_max_idx = serial_find_max(a);
  HRTimer end = HR::now();
  auto duration = duration_cast<microseconds>(end - start).count();
  cout << "Sequential max index: " << s_max_idx << " in " << duration << " us" << endl;

  start = HR::now();
  uint64_t tbb_max_idx = tbb_find_max(a);
  end = HR::now();
  assert(s_max_idx == tbb_max_idx);
  duration = duration_cast<microseconds>(end - start).count();
  cout << "Parallel (TBB) max index in " << tbb_max_idx << " in " << duration << " us" << endl;

  return EXIT_SUCCESS;
}
