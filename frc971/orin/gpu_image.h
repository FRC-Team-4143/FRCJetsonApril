#pragma once

template <typename T>
struct GpuImage {
  typedef T type;
  T *data;
  size_t rows;
  size_t cols;
  // Step is in elements, not bytes.
  size_t step;
};


