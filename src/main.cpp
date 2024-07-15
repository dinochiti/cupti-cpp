#include <iostream>

#include "cuda_cupti.h"
#include <string_view>

using std::cout;

static const size_t NUM_POINTS = 1024;
static const size_t BLOCK_SIZE = 32;
static const size_t STEP_SIZE = 1;
static const int DEVICE_NUMBER = 0;

int main(int argc, const char* argv[]) {
  size_t num_points = NUM_POINTS;
  size_t block_size = BLOCK_SIZE;
  size_t step_size = STEP_SIZE;
  int device_number = DEVICE_NUMBER;
  if (argc > 1) {
    num_points = atoi(argv[1]);
    // a 'k' at the end means kilobytes; quick-and-dirty way to specify lots of points
    const char *last_char = argv[1];
    while (0 != *(last_char + 1)) {
      ++last_char;
    }
    if ('k' == *last_char || 'K' == *last_char) {
      num_points *= 1024;
    }
    // an 'm' at the end means megabytes
    else if ('m' == *last_char || 'M' == *last_char) {
      num_points *= 1024 * 1024;
    }
  }
  if (argc > 2) {
    block_size = atoi(argv[2]);
  }
  if (argc > 3) {
    step_size = atoi(argv[3]);
  }
  if (argc > 4) {
    device_number = atoi(argv[4]);
  }
  // catch excess arguments, invalid (e.g., alpha) arguments
  if (argc > 5 || 0 == num_points || 0 == block_size || 0 == step_size || device_number < 0) {
    cout << "Usage: " << argv[0] << " <number of doubles> <block size> <step size> <device number>\n";
    exit(EXIT_SUCCESS);
  }

  devices_info(num_points * sizeof(double));

  cout << "\nRunning calculations on device " << device_number << "\n";
  cout << "  Points:     " << num_points << "\n";
  cout << "  Block size: " << block_size << "\n";
  cout << "  Step size:  " << step_size << "\n";

  run_calculations(num_points, block_size, step_size, device_number);

  return 0;
}
