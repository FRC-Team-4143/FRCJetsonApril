#include <numeric>
#include <random>
#include <string>

#include "absl/flags/declare.h"
#include "absl/flags/flag.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include <gtest/gtest.h>
#include "opencv2/imgproc.hpp"
//#include "third_party/apriltag/apriltag.h"
#include "third_party/apriltag/tag16h5.h"
#include "third_party/apriltag/tag36h11.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>

#include "aos/time/time.h"
#include "frc971/orin/apriltag.h"

ABSL_FLAG(int32_t, pixel_border, 10,
          "Size of image border within which to reject detected corners");
ABSL_FLAG(double, min_decision_margin, 50.0,
          "Minimum decision margin (confidence) for an apriltag detection");

ABSL_FLAG(bool, debug, false, "If true, write debug images.");

ABSL_DECLARE_FLAG(int32_t, debug_blob_index);

namespace frc971::apriltag::testing {

// Makes a tag detector.
apriltag_detector_t *MakeTagDetector(apriltag_family_t *tag_family) {
  apriltag_detector_t *tag_detector = apriltag_detector_create();

  apriltag_detector_add_family_bits(tag_detector, tag_family, 1);

  tag_detector->nthreads = 6;
  tag_detector->wp = workerpool_create(tag_detector->nthreads);
  tag_detector->qtp.min_white_black_diff = 5;
  tag_detector->debug = absl::GetFlag(FLAGS_debug);

  return tag_detector;
}

// TODO(max): Create a function which will take in the calibration data
CameraMatrix create_camera_matrix() {
  return CameraMatrix{
      1,
      1,
      1,
      1,
  };
}

DistCoeffs create_distortion_coefficients() {
  return DistCoeffs{
      0, 0, 0, 0, 0,
  };
}

class CudaAprilTagDetector {
 public:
  CudaAprilTagDetector(size_t width, size_t height,
                       apriltag_family_t *tag_family = tag16h5_create())
      : tag_family_(tag_family),
	tag_detector_(MakeTagDetector(tag_family_)),
        gray_cuda_(cv::Size(width, height), CV_8UC1),
        decimated_cuda_(gray_cuda_.size() / 2, CV_8UC1),
        thresholded_cuda_(decimated_cuda_.size(), CV_8UC1),
        gpu_detector_(width, height, tag_detector_, create_camera_matrix(),
                      create_distortion_coefficients()),
        width_(width),
        height_(height) {
    // Report out info about our GPU.
    {
      cudaDeviceProp prop;
      CHECK_EQ(cudaGetDeviceProperties(&prop, 0), cudaSuccess);

      LOG(INFO) << "Device: sm_" << prop.major << prop.minor;
#define DUMP(x) LOG(INFO) << "" #x ": " << prop.x;
      DUMP(sharedMemPerBlock);
      DUMP(l2CacheSize);
      DUMP(maxThreadsPerBlock);
      DUMP(maxThreadsPerMultiProcessor);
      DUMP(memoryBusWidth);
      DUMP(memoryClockRate);
      DUMP(multiProcessorCount);
      DUMP(maxBlocksPerMultiProcessor);
      DUMP(name);
      DUMP(warpSize);

#undef DUMP
    }

  }

  ~CudaAprilTagDetector() {
    apriltag_detector_destroy(tag_detector_);
    free(tag_family_);
  }

  // Detects tags on the GPU.
  void DetectGPU(cv::Mat color_image) {
    CHECK_EQ(color_image.size(), gray_cuda_.size());

    gpu_detector_.Detect(color_image.data);

    gpu_detector_.CopyGrayTo(gray_cuda_.data);
    gpu_detector_.CopyDecimatedTo(decimated_cuda_.data);
    gpu_detector_.CopyThresholdedTo(thresholded_cuda_.data);

    extents_cuda_ = gpu_detector_.CopyExtents();
    selected_extents_cuda_ = gpu_detector_.CopySelectedExtents();
    selected_blobs_cuda_ = gpu_detector_.CopySelectedBlobs();
    sorted_selected_blobs_cuda_ = gpu_detector_.CopySortedSelectedBlobs();
    line_fit_points_cuda_ = gpu_detector_.CopyLineFitPoints();
    errors_device_ = gpu_detector_.CopyErrors();
    filtered_errors_device_ = gpu_detector_.CopyFilteredErrors();
    peaks_device_ = gpu_detector_.CopyPeaks();
    num_quads_ = gpu_detector_.NumQuads();

    fit_quads_ = gpu_detector_.FitQuads();

  }

  // Sets the camera constants for camera 24-04
  void SetCameraFourConstants() {
    gpu_detector_.SetCameraMatrix(
        CameraMatrix{642.80365, 718.017517, 642.83667, 555.022461});
    gpu_detector_.SetDistortionCoefficients(
        DistCoeffs{-0.239969, 0.055889, 0.000086, 0.000099, -0.005468});
  }

 private:
  apriltag_family_t *tag_family_;
  apriltag_detector_t *tag_detector_;

  cv::Mat gray_cuda_;
  cv::Mat decimated_cuda_;
  cv::Mat thresholded_cuda_;


  std::vector<uint32_t> quad_length_;
  std::vector<MinMaxExtents> extents_cuda_;
  std::vector<cub::KeyValuePair<long, MinMaxExtents>> selected_extents_cuda_;
  std::vector<IndexPoint> selected_blobs_cuda_;
  std::vector<IndexPoint> sorted_selected_blobs_cuda_;
  std::vector<LineFitPoint> line_fit_points_cuda_;

  std::vector<double> errors_device_;
  std::vector<double> filtered_errors_device_;
  std::vector<Peak> peaks_device_;
  int num_quads_ = 0;
  std::vector<QuadCorners> fit_quads_;

  GpuDetector gpu_detector_;

  size_t width_;
  size_t height_;

  bool normal_border_ = false;
  bool reversed_border_ = false;

  bool undistort_ = false;
  int min_tag_width_ = 1000000;
};

int
main() {
  cv::Mat color_image(1280,720, CV_8UC3);

  CudaAprilTagDetector cuda_detector(color_image.cols,
                                     color_image.rows);

  cuda_detector.DetectGPU(color_image.clone());
  return(0);
}

}

