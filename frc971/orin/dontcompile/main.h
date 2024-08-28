#include <numeric>
#include <random>
#include <string>

#include "opencv2/imgproc.hpp"
#include "third_party/apriltag/apriltag.h"
#include "third_party/apriltag/tag36h11.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>

//#include "aos/time/time.h"
#include "frc971/orin/971apriltag.h"


namespace frc971::apriltag::testing {

// Makes a tag detector.
apriltag_detector_t *MakeTagDetector(apriltag_family_t *tag_family);

class CudaAprilTagDetector {
 public:
  CudaAprilTagDetector(size_t width, size_t height,
                       apriltag_family_t *tag_family);

  virtual ~CudaAprilTagDetector();


  // Detects tags on the GPU.
  void DetectGPU(cv::Mat color_image);

  // Sets the camera constants for camera 24-04
  void SetCameraFourConstants() ;

  int num_quads_ = -1;

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
  std::vector<QuadCorners> fit_quads_;

  GpuDetector gpu_detector_;

  size_t width_;
  size_t height_;

  bool normal_border_;
  bool reversed_border_;

  bool undistort_;
  int min_tag_width_;
};

}


