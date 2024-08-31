#include "frc971/orin/cuda.h"

namespace frc971::apriltag {

size_t overall_memory = 0;

void CheckAndSynchronize(std::string_view message) {
  CHECK_CUDA(cudaDeviceSynchronize()) << message;
  CHECK_CUDA(cudaGetLastError()) << message;
}

void MaybeCheckAndSynchronize() {
  if ( false /*absl::GetFlag(FLAGS_sync) */) CheckAndSynchronize();
}

void MaybeCheckAndSynchronize(std::string_view message) {
  if ( false /*absl::GetFlag(FLAGS_sync) */) CheckAndSynchronize(message);
}

}  // namespace frc971::apriltag
