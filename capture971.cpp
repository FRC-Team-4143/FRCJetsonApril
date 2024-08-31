#include <asm/types.h>
#include <assert.h>
#include <cscore.h>
#include <cscore_cv.h>
#include <cuda_runtime.h>
#include <errno.h>
#include <fcntl.h> 
#include <fmt/format.h>
#include <getopt.h>
#include <linux/videodev2.h>
#include <malloc.h>
#include <networktables/NetworkTableInstance.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include <chrono>
#include <iostream>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "cuAprilTags.h"
// #include <eigen3/Eigen/Dense>

#include "frc/apriltag/AprilTagFieldLayout.h"
#include "frc/apriltag/AprilTagFields.h"
#include "raw2rgb.cuh"

#define OV9281_MIN_GAIN 0x0000
#define OV9281_MAX_GAIN 0x00FE
//#define OV9281_DEFAULT_GAIN 0x0010 /* 1.0x real gain */
#define OV9281_DEFAULT_GAIN 180

#define OV9281_MIN_EXPOSURE_COARSE 0x00000010
#define OV9281_MAX_EXPOSURE_COARSE 0x00003750
//#define OV9281_DEFAULT_EXPOSURE_COARSE 0x00002A90
#define OV9281_DEFAULT_EXPOSURE_COARSE 3847

#define CLEAR(x) memset(&(x), 0, sizeof(x))
#define ARRAY_SIZE(a) (sizeof(a) / sizeof((a)[0]))

#define EIGHTBIT  // otherwise 10bit input

struct buffer {
    void *start;
    size_t length;
};

const int team = 4143;

const int fieldWidth = 1700;
const int fieldHeight = 850;
//100px in a meater

const int csDecimate = 2;

cv::Mat fieldMatEmpty;

// Handle used to interface with the stereo library.
cuAprilTagsHandle april_tags_handle = nullptr;

// Camera intrinsics
// Innovision OV9281rawv2
// theoretical
//cuAprilTagsCameraIntrinsics_t cam_intrinsics =
//    {1082., 1082., 1280.0 / 2.0, 800.0 / 2.0};
//    random camera from Cole
cuAprilTagsCameraIntrinsics_t cam_intrinsics =
    {975.313802, 973.643651, 653.72789, 394.844602};

cv::Mat intrinsicMat(3, 3, cv::DataType<double>::type);
const std::vector<cv::Point3d> objectPoints = {
    cv::Point3d(-.1, .1, 0),
    cv::Point3d(-.1, -.1, 0),
    cv::Point3d(.1, -.1, 0),
    cv::Point3d(.1, .1, 0),
    cv::Point3d(-.1, .1, -.2),
    cv::Point3d(-.1, -.1, -.2),
    cv::Point3d(.1, -.1, -.2),
    cv::Point3d(.1, .1, -.2),
    cv::Point3d(0, 0, 0),
    cv::Point3d(.1, 0, 0),
    cv::Point3d(0, .1, 0),
    cv::Point3d(0, 0, -.1)};
// cv::Mat distCoeffs = (cv::Mat_<double>(5, 1) << -0.4647801828154495, 0.2844034418347612, -0.008834734456932225, -0.01218903939069973, -0.1319188405435461);
// from Cole random camera
cv::Mat distCoeffs = (cv::Mat_<double>(5, 1) << -0.307453, 0.106324, -0.004022, -0.006423, 0.0);
//cv::Mat distCoeffs = (cv::Mat_<double>(5, 1) << 0.0, 0.0, 0.0, 0.0, 0.0);

// Output vector of detected Tags
std::vector<cuAprilTagsID_t> tags;

// CUDA stream
cudaStream_t main_stream = {};

const float tag_size = .16;  // same units as camera calib
#define HALFTAGSIZE .08_m
const int max_tags = 4;
const int tile_size = 24;

cuAprilTagsImageInput_t input_image;

#ifndef DEVICE
#define DEVICE 0
#endif

#if DEVICE == 1
static const char *dev_name = "/dev/video1";
#else
static const char *dev_name = "/dev/video0";
#endif

static int fd = -1;
struct buffer *buffers = NULL;
static unsigned int n_buffers = 0;
static unsigned int width = 1280;
static unsigned int height = 720;
static unsigned int count = 0;
static unsigned char *cuda_out_buffer = NULL;
#ifdef EIGHTBIT
static unsigned int pixel_format = V4L2_PIX_FMT_SRGGB8;
#else
static unsigned int pixel_format = V4L2_PIX_FMT_SRGGB10;
#endif
static unsigned int field = V4L2_FIELD_NONE;


#include "frc971/orin/971apriltag.h"
#include "third_party/apriltag/apriltag.h"
#include "third_party/apriltag/tag36h11.h"

// TODO(max): Create a function which will take in the calibration data
frc971::apriltag::CameraMatrix create_camera_matrix() {
  return frc971::apriltag::CameraMatrix{
      1,
      1,
      1,
      1,
  };
}

frc971::apriltag::DistCoeffs create_distortion_coefficients() {
  return frc971::apriltag::DistCoeffs{
      0, 0, 0, 0, 0,
  };
}

// Makes a tag detector.
apriltag_detector_t *MakeTagDetector(apriltag_family_t *tag_family) {
  apriltag_detector_t *tag_detector = apriltag_detector_create();

  apriltag_detector_add_family_bits(tag_detector, tag_family, 1);

  tag_detector->nthreads = 6;
  tag_detector->wp = workerpool_create(tag_detector->nthreads);
  tag_detector->qtp.min_white_black_diff = 5;
  tag_detector->debug = false;

  return tag_detector;
}

frc971::apriltag::GpuDetector *gpu_detector_;

void drawAprilBoundingBox971(apriltag_detection_t * detection, cv::Mat &mat);


auto ntinst = nt::NetworkTableInstance::GetDefault();

cs::CvSource cvsource{"cvsource", cs::VideoMode::kMJPEG, (int)width/csDecimate, (int)height/csDecimate, 8};
cs::CvSource cvsource2{"cvsource2", cs::VideoMode::kMJPEG, (int)fieldHeight/csDecimate, (int)fieldWidth/csDecimate, 8};
#if DEVICE == 1
cs::MjpegServer cvMjpegServer{dev_name, 1183};
cs::MjpegServer cvMjpegServer2{"field", 1184};
#else
cs::MjpegServer cvMjpegServer{dev_name, 1181};
cs::MjpegServer cvMjpegServer2{"field", 1182};
#endif

frc::AprilTagFieldLayout fieldlayout;

const std::string_view config = "{ }";

void matrixToRPY(const float *matrix, float &roll, float &pitch, float &yaw);
void drawField(cv::Mat &fieldMat);
void drawAprilBoundingBox(cuAprilTagsID_t detections, cv::Mat &mat);
float tagDist(cuAprilTagsID_t tag);
frc::Translation3d triangulate(cuAprilTagsID_t tag, cuAprilTagsID_t tagMinusOne);
float lawOfCosines(float a, float b, float c);
void print_pose3d(frc::Pose3d pos, std::string description);
frc::Pose3d get_field_tag_corner(frc::Pose3d, int corner);

static void
errno_exit(const char *s) {
    fprintf(stderr, "%s error %d, %s\n",
            s, errno, strerror(errno));

    exit(EXIT_FAILURE);
}

static int
xioctl(int fd,
       int request,
       void *arg) {
    int r;

    do
        r = ioctl(fd, request, arg);
    while (-1 == r && EINTR == errno);

    return r;
}

// 0 == Pitch 1 == Roll 2 == Yaw 
cv::Vec3f rotationMatrixToEulerAngles(cv::Mat &R) {
    //assert(isRotationMatrix(R));
    float sy = sqrt(R.at<double>(0,0) * R.at<double>(0,0) + R.at<double> (1,0) * R.at<double> (1,0));
    bool singular = sy < 1e-6;
    float x, y, z;
    if (!singular) {
        x = atan2(R.at<double>(2,1), R.at<double>(2,2));
        y = atan2(-R.at<double>(2,0), sy);
        z = atan2(R.at<double>(1,0), R.at<double>(0,0));
    } else {
        x = atan2(-R.at<double>(1,2), R.at<double>(1,1));
        y = atan2(-R.at<double>(2,0), sy);
        z = 0;
    }
    return cv::Vec3f(x, y, z);
}

static void
process_image(void *p, double fps) {
	gpu_detector_->DetectGray((unsigned char *)p);

        const zarray_t *detections = gpu_detector_->Detections();
	if( zarray_size(detections))
	for (int i = 0; i < zarray_size(detections); ++i) {
	     std::cout << "971 library has " << zarray_size(detections) << " detections" << std::endl;
             apriltag_detection_t *gpu_detection;
             zarray_get(detections, i, &gpu_detection);
	     std::cout << "tag: " << gpu_detection->id << " at " <<
		     gpu_detection->c[0] << "," << gpu_detection->c[1] 
		     << std::endl;
	}

    // output to cameraserver every 10 frames
    if (count % 10 == 0) {

#ifdef EIGHTBIT
    gpuConvertgraytoRGB((unsigned char *)p, cuda_out_buffer, width, height, main_stream);
#else
    gpuConvertgraytoRGB((unsigned short *)p, cuda_out_buffer, width, height, main_stream);
#endif
    cudaStreamAttachMemAsync(main_stream, cuda_out_buffer, 0, cudaMemAttachHost);
    cudaStreamSynchronize(main_stream);

#undef NVAPRILTAG
#ifdef NVAPRILTAG
    uint32_t num_detections;
    input_image.dev_ptr = (uchar3 *)cuda_out_buffer;
    input_image.pitch = width * 3;
    // cudaStreamAttachMemAsync(main_stream, input_image.dev_ptr, 0, cudaMemAttachGlobal);
    const int error = cuAprilTagsDetect(
        april_tags_handle, &input_image, tags.data(),
        &num_detections, max_tags, main_stream);
    // cudaStreamAttachMemAsync(main_stream, input_image.dev_ptr, 0, cudaMemAttachHost);
    // cudaStreamSynchronize(main_stream);

    if (error != 0) {
        std::cout << "april tag detect error" << std::endl;
    }

    // run solvePnP on all tag corners
  std::vector<cv::Point2d> imagePoints;
  std::vector<cv::Point3d> objectPoints;
  cv::Mat rVec(3, 1, cv::DataType<double>::type, 0.0);
  cv::Mat tVec(3, 1, cv::DataType<double>::type, 0.0);
  cv::Mat R;
  cv::Vec3f EulerAngles;

    for (uint32_t i = 0; i < num_detections; i++) {
        const cuAprilTagsID_t &detection = tags[i];
	
	auto tagPose =  fieldlayout.GetTagPose(detection.id).value_or(frc::Pose3d());

	    for(int i = 0; i < 4; i++){
		imagePoints.push_back(cv::Point(detection.corners[i].x, detection.corners[i].y));
		auto corner = get_field_tag_corner(tagPose, i);
		objectPoints.push_back(cv::Point3d(corner.X().value(), corner.Y().value(), corner.Z().value()));
	    }
    }

#if DEVICE == 1
    auto table = ntinst.GetTable("WarVision1");
#else
    auto table = ntinst.GetTable("WarVision");
#endif

    table->PutNumber("numdetections", num_detections);
    if(num_detections > 0) {
	cv::solvePnP(objectPoints, imagePoints, intrinsicMat, distCoeffs, rVec, tVec, false, cv::SOLVEPNP_SQPNP);

  	// std::cout << "tVec " << tVec << std::endl;
	// std::cout << "rVec " << rVec << std::endl;
	// inverse the projections from world to camera
	cv::Rodrigues(rVec, R);
	R = R.t();
	tVec = -R * tVec;
        EulerAngles = rotationMatrixToEulerAngles(R);
	//std::cout << "camera position X:" << tVec.at<double>(0) << " Y:" << tVec.at<double>(1) << " Z:" << tVec.at<double>(2) << std::endl;
        //std::cout << "EulerAngles X:" << EulerAngles[0] << " Y:" << EulerAngles[1] << " Z:"<< EulerAngles[2] << std::endl;
            
        table->PutNumber("botposeX", tVec.at<double>(0));
        table->PutNumber("botposeY", tVec.at<double>(1));
        table->PutNumber("botposeZ", tVec.at<double>(2));
        table->PutNumber("EulerAngleX", EulerAngles[0]);
        table->PutNumber("EulerAngleY", EulerAngles[1]);
    	table->PutNumber("EulerAngleZ", EulerAngles[2]);
    }

    // print radians between multiple tags
    /*
        if (num_detections > 1) {
            for (uint32_t i = 1; i < num_detections; i++) {
		    std::cout << sqrt(pow(tags[i].corners[0].x-tags[i-1].corners[0].x,2) + pow(tags[i].corners[0].y-tags[i-1].corners[0].y,2)) / cam_intrinsics.fx << " radians apart" << std::endl;
	    }
        } 
    */
#endif

        cv::Mat mat(height, width, CV_8UC3, cuda_out_buffer);
        cv::Mat fieldMat = fieldMatEmpty.clone();

	for (int i = 0; i < zarray_size(detections); ++i) {
             apriltag_detection_t *gpu_detection;
             zarray_get(detections, i, &gpu_detection);
             drawAprilBoundingBox971(gpu_detection, mat);
	}

#ifdef NVAPRILTAG
        for (uint32_t i = 0; i < num_detections; i++) {
            const cuAprilTagsID_t &detection = tags[i];

            float roll, pitch, yaw;
            matrixToRPY(detection.orientation, pitch, yaw, roll);

            //std::cout << yaw << std::endl;

            drawAprilBoundingBox(detection, mat);

            auto tagTranslation = frc::Translation3d(units::meter_t(detection.translation[2]), units::meter_t(-detection.translation[0]), units::meter_t(0));
            auto tagTransform = frc::Transform3d(tagTranslation, frc::Rotation3d());  // put in pitch
            auto tagTranslationRot = frc::Translation3d();
            auto tagTransformRot = frc::Transform3d(tagTranslationRot, frc::Rotation3d(units::radian_t(0), units::radian_t(0), units::radian_t(-yaw)));  // put in pitch

	    auto tagPose =  fieldlayout.GetTagPose(detection.id).value_or(frc::Pose3d());
            auto robotPose = tagPose + tagTransformRot + tagTransform;
            cv::circle(fieldMat, cv::Point(robotPose.X().value() * 100, fieldHeight - robotPose.Y().value() * 100), 20, cv::Scalar(50 + 200 * (detection.id-1 & 1), 50 + 200 * (detection.id-1 >> 1 & 1), 50 + 200 * (detection.id-1 >> 2 & 1)), 2);


        }

    	if(num_detections > 0) {
		// print solvePnP white dot
		cv::circle(fieldMat, cv::Point(tVec.at<double>(0) * 100, fieldHeight - tVec.at<double>(1) * 100), 10, cv::Scalar(255,255,255), 20);

    		cv::line(fieldMat, cv::Point(tVec.at<double>(0) * 100, fieldHeight - tVec.at<double>(1) * 100), cv::Point((tVec.at<double>(0)-sin(EulerAngles[2])) * 100, fieldHeight - (tVec.at<double>(1)+cos(EulerAngles[2])) * 100), cv::Scalar(255, 255, 255), 5);
	
        	cv::putText(fieldMat, std::to_string(num_detections), cv::Point(tVec.at<double>(0) * 100 - 10, fieldHeight - tVec.at<double>(1) * 100 + 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0,0,0), 2, false);
	}
#endif

        std::string str = "fps: " + std::to_string(fps);
        cv::putText(mat, str, cv::Point(50, 50), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(255, 0, 0), 2, false);

	cv::resize(mat,mat,cv::Size(mat.cols/csDecimate, mat.rows/csDecimate), 0, 0, cv::INTER_NEAREST);
        cvsource.PutFrame(mat);

	cv::resize(fieldMat,fieldMat,cv::Size(fieldMat.cols/csDecimate, fieldMat.rows/csDecimate), 0, 0, cv::INTER_NEAREST);
        cvsource2.PutFrame(fieldMat);
    }
}

void matrixToRPY(const float *matrix, float &roll, float &pitch, float &yaw) {
    yaw = atan2(matrix[1], matrix[0]);
    pitch = atan2(matrix[2], sqrt(pow(matrix[5], 2) + pow(matrix[8], 2)));
    roll = atan2(matrix[5], matrix[8]);
}

void drawField(cv::Mat &fieldMat) {
    // draw field need const
    cv::line(fieldMat, cv::Point(5, 5), cv::Point(fieldWidth-10, 5), cv::Scalar(0, 255, 0), 5);
    cv::line(fieldMat, cv::Point(5, fieldHeight-5), cv::Point(fieldWidth-10, fieldHeight-5), cv::Scalar(0, 255, 0), 5);
    cv::line(fieldMat, cv::Point(fieldHeight, 0), cv::Point(fieldHeight, fieldHeight), cv::Scalar(0, 255, 0), 2);
    cv::line(fieldMat, cv::Point(5, 5), cv::Point(5, fieldHeight-5), cv::Scalar(255, 0, 0), 5);
    cv::line(fieldMat, cv::Point(fieldWidth-5, 5), cv::Point(fieldWidth-5, fieldHeight-5), cv::Scalar(0, 0, 255), 5);
    
    //wing lines 5.876meters from wall
    cv::line(fieldMat, cv::Point(587, 5), cv::Point(587, fieldHeight-5), cv::Scalar(255, 0, 0), 5);
    cv::line(fieldMat, cv::Point(fieldWidth-587, 5), cv::Point(fieldWidth-587, fieldHeight-5), cv::Scalar(0, 0, 255), 5);

    //amp zones 3.302meters from wall 0.4572meter from other wall
    cv::line(fieldMat, cv::Point(fieldWidth-5, 45), cv::Point(fieldWidth-330, 45), cv::Scalar(0, 0, 255), 5);
    cv::line(fieldMat, cv::Point(fieldWidth-330, 5), cv::Point(fieldWidth-330, 45), cv::Scalar(0, 0, 255), 5);
    cv::line(fieldMat, cv::Point(330, 5), cv::Point(330, 45), cv::Scalar(255, 0, 0), 5);
    cv::line(fieldMat, cv::Point(5, 45), cv::Point(330, 45), cv::Scalar(255, 0, 0), 5);

    //stage zones 3.073meters from wall 3.115meters wide
    cv::line(fieldMat, cv::Point(307, fieldHeight/2), cv::Point(587, fieldHeight/2 + 155), cv::Scalar(255, 0, 0), 5);
    cv::line(fieldMat, cv::Point(307, fieldHeight/2), cv::Point(587, fieldHeight/2 - 155), cv::Scalar(255, 0, 0), 5);
    cv::line(fieldMat, cv::Point(fieldWidth-307, fieldHeight/2), cv::Point(fieldWidth-587, fieldHeight/2 + 155), cv::Scalar(0, 0, 255), 5);
    cv::line(fieldMat, cv::Point(fieldWidth-307, fieldHeight/2), cv::Point(fieldWidth-587, fieldHeight/2 - 155), cv::Scalar(0, 0, 255), 5);


    for (int i = 1; i < 17; i++) {
        auto pos = fieldlayout.GetTagPose(i).value_or(frc::Pose3d());
        cv::putText(fieldMat, std::to_string(i), cv::Point(pos.X().value() * 100, fieldHeight-pos.Y().value() * 100), cv::FONT_HERSHEY_DUPLEX, 2, cv::Scalar(50 + 200 * (i-1 & 1), 50 + 200 * (i-1 >> 1 & 1), 50 + 200 * (i-1 >> 2 & 1)), 2, false);
    }
}

void drawAprilBoundingBox971(apriltag_detection_t * detection, cv::Mat &mat) {
    cv::Mat rRot(3, 3, cv::DataType<double>::type, 0.0);
    cv::Mat rVec(3, 1, cv::DataType<double>::type, 0.0);
    cv::Mat tVec(3, 1, cv::DataType<double>::type, 0.0);
    cv::Point tag_points[4];
    cv::Point center_point;
    std::vector<cv::Point2d> imagePoints;
    tag_points[0] = cv::Point(detection->p[0][0], detection->p[0][1]);
    tag_points[1] = cv::Point(detection->p[1][0], detection->p[1][1]);
    tag_points[2] = cv::Point(detection->p[2][0], detection->p[2][1]);
    tag_points[3] = cv::Point(detection->p[3][0], detection->p[3][1]);
    cv::line(mat, tag_points[0], tag_points[1], cv::Scalar(255, 255, 0), 4);
    cv::line(mat, tag_points[1], tag_points[2], cv::Scalar(255, 255, 0), 4);
    cv::line(mat, tag_points[2], tag_points[3], cv::Scalar(255, 255, 0), 4);
    cv::line(mat, tag_points[3], tag_points[0], cv::Scalar(255, 255, 0), 4);
    center_point = cv::Point(detection->c[0], detection->c[1]);

    /*
    for (int i = 0; i < 9; i++)
        rRot.at<double>(i) = detection->.H[i];
    cv::transpose(rRot, rRot);
    cv::Rodrigues(rRot, rVec);

    tVec.at<double>(0) = detection->.translation[0]; //don't have yet RJS
    tVec.at<double>(1) = detection->.translation[1];
    tVec.at<double>(2) = detection->.translation[2];

    // std::cout << "Intrisic matrix: " << intrinsicMat << std::endl << std::endl;
    // std::cout << "Rotation matrix: " << rRot << std::endl << std::endl;
    // std::cout << "Rotation vector: " << rVec << std::endl << std::endl;
    // std::cout << "Translation vector: " << tVec << std::endl << std::endl;
    cv::projectPoints(objectPoints, rVec, tVec, intrinsicMat, distCoeffs, imagePoints);
    // for (unsigned int i = 0; i < imagePoints.size(); ++i){
    //   cv::circle(mat, imagePoints[i], 3, cv::Scalar(0,255,0), 1);
    // }
    cv::line(mat, imagePoints[0], imagePoints[1], cv::Scalar(0, 255, 0), 2);  // Box stuff
    cv::line(mat, imagePoints[1], imagePoints[2], cv::Scalar(0, 255, 0), 2);
    cv::line(mat, imagePoints[2], imagePoints[3], cv::Scalar(0, 255, 0), 2);
    cv::line(mat, imagePoints[3], imagePoints[0], cv::Scalar(0, 255, 0), 2);
    cv::line(mat, imagePoints[4], imagePoints[5], cv::Scalar(0, 255, 0), 2);
    cv::line(mat, imagePoints[5], imagePoints[6], cv::Scalar(0, 255, 0), 2);
    cv::line(mat, imagePoints[6], imagePoints[7], cv::Scalar(0, 255, 0), 2);
    cv::line(mat, imagePoints[7], imagePoints[4], cv::Scalar(0, 255, 0), 2);
    cv::line(mat, imagePoints[0], imagePoints[4], cv::Scalar(0, 255, 0), 2);
    cv::line(mat, imagePoints[1], imagePoints[5], cv::Scalar(0, 255, 0), 2);
    cv::line(mat, imagePoints[2], imagePoints[6], cv::Scalar(0, 255, 0), 2);
    cv::line(mat, imagePoints[3], imagePoints[7], cv::Scalar(0, 255, 0), 2);
    cv::line(mat, imagePoints[8], imagePoints[9], cv::Scalar(255, 0, 0), 2);  // Next three are coordinate pointers
    cv::line(mat, imagePoints[8], imagePoints[10], cv::Scalar(0, 255, 0), 2);
    cv::line(mat, imagePoints[8], imagePoints[11], cv::Scalar(0, 0, 255), 2);
    */
    std::string str = std::to_string(detection->id);
    cv::putText(mat, str, center_point, cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 255, 255), 2, false);
    cv::putText(mat, "0", tag_points[0], cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(255, 0, 255), 2, false);
    cv::putText(mat, "1", tag_points[1], cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(255, 0, 255), 2, false);
    cv::putText(mat, "2", tag_points[2], cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(255, 0, 255), 2, false);
    cv::putText(mat, "3", tag_points[3], cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(255, 0, 255), 2, false);
}

void drawAprilBoundingBox(cuAprilTagsID_t detection, cv::Mat &mat) {
    cv::Mat rRot(3, 3, cv::DataType<double>::type, 0.0);
    cv::Mat rVec(3, 1, cv::DataType<double>::type, 0.0);
    cv::Mat tVec(3, 1, cv::DataType<double>::type, 0.0);
    cv::Point tag_points[4];
    std::vector<cv::Point2d> imagePoints;
    tag_points[0] = cv::Point(detection.corners[0].x, detection.corners[0].y);
    tag_points[1] = cv::Point(detection.corners[1].x, detection.corners[1].y);
    tag_points[2] = cv::Point(detection.corners[2].x, detection.corners[2].y);
    tag_points[3] = cv::Point(detection.corners[3].x, detection.corners[3].y);
    cv::line(mat, tag_points[0], tag_points[1], cv::Scalar(255, 0, 0), 1);
    cv::line(mat, tag_points[1], tag_points[2], cv::Scalar(255, 0, 0), 1);
    cv::line(mat, tag_points[2], tag_points[3], cv::Scalar(255, 0, 0), 1);
    cv::line(mat, tag_points[3], tag_points[0], cv::Scalar(255, 0, 0), 1);

    for (int i = 0; i < 9; i++)
        rRot.at<double>(i) = detection.orientation[i];
    cv::transpose(rRot, rRot);
    cv::Rodrigues(rRot, rVec);

    tVec.at<double>(0) = detection.translation[0];
    tVec.at<double>(1) = detection.translation[1];
    tVec.at<double>(2) = detection.translation[2];

    // std::cout << "Intrisic matrix: " << intrinsicMat << std::endl << std::endl;
    // std::cout << "Rotation matrix: " << rRot << std::endl << std::endl;
    // std::cout << "Rotation vector: " << rVec << std::endl << std::endl;
    // std::cout << "Translation vector: " << tVec << std::endl << std::endl;
    cv::projectPoints(objectPoints, rVec, tVec, intrinsicMat, distCoeffs, imagePoints);
    // for (unsigned int i = 0; i < imagePoints.size(); ++i){
    //   cv::circle(mat, imagePoints[i], 3, cv::Scalar(0,255,0), 1);
    // }
    cv::line(mat, imagePoints[0], imagePoints[1], cv::Scalar(0, 255, 0), 2);  // Box stuff
    cv::line(mat, imagePoints[1], imagePoints[2], cv::Scalar(0, 255, 0), 2);
    cv::line(mat, imagePoints[2], imagePoints[3], cv::Scalar(0, 255, 0), 2);
    cv::line(mat, imagePoints[3], imagePoints[0], cv::Scalar(0, 255, 0), 2);
    cv::line(mat, imagePoints[4], imagePoints[5], cv::Scalar(0, 255, 0), 2);
    cv::line(mat, imagePoints[5], imagePoints[6], cv::Scalar(0, 255, 0), 2);
    cv::line(mat, imagePoints[6], imagePoints[7], cv::Scalar(0, 255, 0), 2);
    cv::line(mat, imagePoints[7], imagePoints[4], cv::Scalar(0, 255, 0), 2);
    cv::line(mat, imagePoints[0], imagePoints[4], cv::Scalar(0, 255, 0), 2);
    cv::line(mat, imagePoints[1], imagePoints[5], cv::Scalar(0, 255, 0), 2);
    cv::line(mat, imagePoints[2], imagePoints[6], cv::Scalar(0, 255, 0), 2);
    cv::line(mat, imagePoints[3], imagePoints[7], cv::Scalar(0, 255, 0), 2);
    cv::line(mat, imagePoints[8], imagePoints[9], cv::Scalar(255, 0, 0), 2);  // Next three are coordinate pointers
    cv::line(mat, imagePoints[8], imagePoints[10], cv::Scalar(0, 255, 0), 2);
    cv::line(mat, imagePoints[8], imagePoints[11], cv::Scalar(0, 0, 255), 2);
    std::string str = std::to_string(detection.id);
    cv::putText(mat, str, imagePoints[8], cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 255, 255), 2, false);
    cv::putText(mat, "0", tag_points[0], cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(255, 0, 255), 2, false);
    cv::putText(mat, "1", tag_points[1], cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(255, 0, 255), 2, false);
    cv::putText(mat, "2", tag_points[2], cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(255, 0, 255), 2, false);
    cv::putText(mat, "3", tag_points[3], cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(255, 0, 255), 2, false);
}

float tagDist(cuAprilTagsID_t tag) {
    return sqrt(tag.translation[0] * tag.translation[0] + tag.translation[2] * tag.translation[2]);
}

frc::Translation3d triangulate(cuAprilTagsID_t tag, cuAprilTagsID_t tagMinusOne) {
    auto iTagPos = fieldlayout.GetTagPose(tag.id).value_or(frc::Pose3d());
    auto iTagPosMinusOne = fieldlayout.GetTagPose(tagMinusOne.id).value_or(frc::Pose3d());
    //float tagDistance = (iTagPos - iTagPosMinusOne).Norm();
    //auto camToI = frc::Translation3d(units::meter_t(tag.translation[2]), units::meter_t(-tag.translation[0]), units::meter_t(0));
    //auto camToIMinusOne = frc::Translation3d(units::meter_t(tagMinusOne.translation[2]), units::meter_t(-tagMinusOne.translation[0]), units::meter_t(0));
    //float iTagDistance = camToI.Norm();
    //float iTagDistanceMinusOne = camToIMinusOne.Norm();
    //float gammaI = lawOfCosines(tagDistance, iTagDistance, iTagDistanceMinusOne);
    //float gammaIMinusOne = lawOfCosines(tagDistance, iTagDistanceMinusOne, iTagDistance);
    //auto iEstimatedTag = frc::Translation3d(units::meter_t(iTagDistance), units::meter_t(0), units::meter_t(0))
                             //.RotateBy(frc::);
    //auto robotPos1 = iTagPos.Translation() +
    return frc::Translation3d();
}

float lawOfCosines(float a, float b, float c) {
    return acos((a * a + b * b - c * c) / (2 * a * b));
}

static int
read_frame(double fps) {
    struct v4l2_buffer buf;
    unsigned int i;

    CLEAR(buf);

    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_USERPTR;

    auto start = std::chrono::high_resolution_clock::now();
    if (-1 == xioctl(fd, VIDIOC_DQBUF, &buf)) {
        switch (errno) {
            case EAGAIN:
                return 0;

            case EIO:
                /* Could ignore EIO, see spec. */

                /* fall through */

            default:
                errno_exit("VIDIOC_DQBUF");
        }
    }

    for (i = 0; i < n_buffers; ++i)
        if (buf.m.userptr == (unsigned long)buffers[i].start && buf.length == buffers[i].length)
            break;

    assert(i < n_buffers);

    fd_set fds;
    struct timeval tv;
    int r;

    FD_ZERO(&fds);
    FD_SET(fd, &fds);

    /* Timeout. */
    tv.tv_sec = 0;
    tv.tv_usec = 0;

    r = select(fd + 1, &fds, NULL, NULL, &tv);

    if (r == 0) {
        // std::cout << "buffer timestamp " << buf.timestamp.tv_sec << " " << buf.timestamp.tv_usec << std::endl;
        process_image((void *)buf.m.userptr, fps);
    } else {
        // std::cout << "another frame available skipping this one" << std::endl;
    }

    if (-1 == xioctl(fd, VIDIOC_QBUF, &buf))
        errno_exit("VIDIOC_QBUF");

    if (r == 0) {
        // auto end = std::chrono::high_resolution_clock::now();
        // double milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        // std::cout << "latency (ms): " << milliseconds << std::endl;
        return 1;
    } else
        return 0;
}

static void
mainloop(void) {
    double fps = 0;

    while (true) {
        count = 0;
        auto start = std::chrono::high_resolution_clock::now();
        while (count < 120) {
            for (;;) {
                fd_set fds;
                struct timeval tv;
                int r;

                FD_ZERO(&fds);
                FD_SET(fd, &fds);

                /* Timeout. */
                tv.tv_sec = 1;
                tv.tv_usec = 0;

                r = select(fd + 1, &fds, NULL, NULL, &tv);

                if (-1 == r) {
                    if (EINTR == errno)
                        continue;

                    errno_exit("select");
                }

                if (0 == r) {
                    fprintf(stderr, "select timeout\n");
                    exit(EXIT_FAILURE);
                }

                if (read_frame(fps)) {
                    count++;
                    break;
                }

                /* EAGAIN - continue select loop. */
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        double seconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0;
        fps = ((double)count) / seconds;

        std::string str = "fps: " + std::to_string(fps) + " " + std::to_string(count) + "/" + std::to_string(seconds);
        std::cout << str << std::endl;
    }
}

static void
stop_capturing(void) {
    enum v4l2_buf_type type;

    type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

    if (-1 == xioctl(fd, VIDIOC_STREAMOFF, &type))
        errno_exit("VIDIOC_STREAMOFF");
}

static void
start_capturing(void) {
    unsigned int i;
    enum v4l2_buf_type type;

    for (i = 0; i < n_buffers; ++i) {
        struct v4l2_buffer buf;

        CLEAR(buf);

        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_USERPTR;
        buf.index = i;
        buf.m.userptr = (unsigned long)buffers[i].start;
        buf.length = buffers[i].length;

        if (-1 == xioctl(fd, VIDIOC_QBUF, &buf))
            errno_exit("VIDIOC_QBUF");
    }

    type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

    if (-1 == xioctl(fd, VIDIOC_STREAMON, &type))
        errno_exit("VIDIOC_STREAMON");
}

static void
uninit_device(void) {
    unsigned int i;

    for (i = 0; i < n_buffers; ++i) {
        cudaFree(buffers[i].start);
    }

    free(buffers);

    cudaFree(cuda_out_buffer);
}

static void
init_userp(unsigned int buffer_size) {
    struct v4l2_requestbuffers req;
    unsigned int page_size;

    page_size = getpagesize();
    buffer_size = (buffer_size + page_size - 1) & ~(page_size - 1);

    CLEAR(req);

    req.count = 4;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_USERPTR;

    if (-1 == xioctl(fd, VIDIOC_REQBUFS, &req)) {
        if (EINVAL == errno) {
            fprintf(stderr,
                    "%s does not support "
                    "user pointer i/o\n",
                    dev_name);
            exit(EXIT_FAILURE);
        } else {
            errno_exit("VIDIOC_REQBUFS");
        }
    }

    buffers = (struct buffer *)calloc(4, sizeof(*buffers));

    if (!buffers) {
        fprintf(stderr, "Out of memory\n");
        exit(EXIT_FAILURE);
    }

    for (n_buffers = 0; n_buffers < 4; ++n_buffers) {
        buffers[n_buffers].length = buffer_size;
        cudaMallocManaged(&buffers[n_buffers].start, buffer_size, cudaMemAttachGlobal);

        if (!buffers[n_buffers].start) {
            fprintf(stderr, "Out of memory\n");
            exit(EXIT_FAILURE);
        }
    }
}

static void
setgain(int gain) {
    struct v4l2_control control;
    struct v4l2_ext_controls ext_controls;
    struct v4l2_ext_control ext_control;

    CLEAR(ext_controls);
    CLEAR(ext_control);
    ext_controls.ctrl_class = V4L2_CTRL_ID2CLASS(0x009a2000);
    ext_controls.count = 1;
    ext_controls.controls = &ext_control;

    ext_control.id = 0x009a2009;  // gain
    ext_control.value64 = gain;
    if (-1 == xioctl(fd, VIDIOC_S_EXT_CTRLS, &ext_controls)) {
        errno_exit("VIDIOC_S_CTRL gain");
    }
}

static void
setexposure(int exposure) {
    struct v4l2_control control;
    struct v4l2_ext_controls ext_controls;
    struct v4l2_ext_control ext_control;

    CLEAR(ext_controls);
    CLEAR(ext_control);
    ext_controls.ctrl_class = V4L2_CTRL_ID2CLASS(0x009a2000);
    ext_controls.count = 1;
    ext_controls.controls = &ext_control;

    ext_control.id = 0x009a200a;  // exposure
    ext_control.value64 = exposure;
    if (-1 == xioctl(fd, VIDIOC_S_EXT_CTRLS, &ext_controls)) {
        errno_exit("VIDIOC_S_CTRL exposure");
    }
}

static void
init_device(void) {
    struct v4l2_capability cap;
    struct v4l2_cropcap cropcap;
    struct v4l2_crop crop;
    struct v4l2_format fmt;
    struct v4l2_control control;
    struct v4l2_ext_controls ext_controls;
    struct v4l2_ext_control ext_control;
    unsigned int min;

    if (-1 == xioctl(fd, VIDIOC_QUERYCAP, &cap)) {
        if (EINVAL == errno) {
            fprintf(stderr, "%s is no V4L2 device\n",
                    dev_name);
            exit(EXIT_FAILURE);
        } else {
            errno_exit("VIDIOC_QUERYCAP");
        }
    }

    if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
        fprintf(stderr, "%s is no video capture device\n",
                dev_name);
        exit(EXIT_FAILURE);
    }

    if (!(cap.capabilities & V4L2_CAP_STREAMING)) {
        fprintf(stderr, "%s does not support streaming i/o\n",
                dev_name);
        exit(EXIT_FAILURE);
    }

    CLEAR(control);
    control.id = 0x009a2064;  // BYPASS_MODE
    control.value = 0;
    if (-1 == xioctl(fd, VIDIOC_S_CTRL, &control)) {
        errno_exit("VIDIOC_S_CTRL bypass mode");
    }

    CLEAR(control);
    control.id = 0x009a206d;  // low_latency_mode
    control.value = 1;
    if (-1 == xioctl(fd, VIDIOC_S_CTRL, &control)) {
        errno_exit("VIDIOC_S_CTRL low latency mode");
    }

    CLEAR(ext_controls);
    CLEAR(ext_control);
    ext_controls.ctrl_class = V4L2_CTRL_ID2CLASS(0x009a2000);
    ext_controls.count = 1;
    ext_controls.controls = &ext_control;

    //    ext_control.id = 0x009a2008; // sensor_mode
    //    ext_control.value64 = 0;
    //    ext_control.value64 = 1;
    //    ext_control.value64 = 4;
    //    ext_control.value64 = 3;
    //    if (-1 == xioctl (fd, VIDIOC_S_EXT_CTRLS, &ext_controls)) {
    //        errno_exit ("VIDIOC_S_CTRL sensor mode");
    //    }

    //    ext_control.id = 0x009a200b; // FRAME_RATE
    //    ext_control.value64 = 120000000;
    //    if (-1 == xioctl (fd, VIDIOC_S_EXT_CTRLS, &ext_controls)) {
    //        errno_exit ("VIDIOC_S_CTRL frame rate");
    //    }

    ext_control.id = 0x009a2009;  // gain
    ext_control.value64 = OV9281_DEFAULT_GAIN;
    if (-1 == xioctl(fd, VIDIOC_S_EXT_CTRLS, &ext_controls)) {
        errno_exit("VIDIOC_S_CTRL gain");
    }

    ext_control.id = 0x009a200a;  // coarse exposure
    ext_control.value64 = OV9281_DEFAULT_EXPOSURE_COARSE;
    if (-1 == xioctl(fd, VIDIOC_S_EXT_CTRLS, &ext_controls)) {
        errno_exit("VIDIOC_S_CTRL exposure");
    }

    /* Select video input, video standard and tune here. */

#if 0
    CLEAR (cropcap);

    cropcap.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

    if (0 == xioctl (fd, VIDIOC_CROPCAP, &cropcap)) {
        crop.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        crop.c = cropcap.defrect; /* reset to default */

        if (-1 == xioctl (fd, VIDIOC_S_CROP, &crop)) {
            switch (errno) {
                case EINVAL:
                    /* Cropping not supported. */
                    break;
                default:
                    /* Errors ignored. */
                    break;
            }
        }
    } else {
        /* Errors ignored. */
    }
#endif

    CLEAR(fmt);

    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = width;
    fmt.fmt.pix.height = height;
    fmt.fmt.pix.pixelformat = pixel_format;
    fmt.fmt.pix.field = field;

    if (-1 == xioctl(fd, VIDIOC_S_FMT, &fmt))
        errno_exit("VIDIOC_S_FMT");

    /* Note VIDIOC_S_FMT may change width and height. */
    fprintf(stderr, "width %d height %d bytesperline %d sizeimage %d\n",
            fmt.fmt.pix.width, fmt.fmt.pix.height,
            fmt.fmt.pix.bytesperline, fmt.fmt.pix.sizeimage);

    /* Buggy driver paranoia. */
    min = fmt.fmt.pix.width * 2;
    if (fmt.fmt.pix.bytesperline < min)
        fmt.fmt.pix.bytesperline = min;
    min = fmt.fmt.pix.bytesperline * fmt.fmt.pix.height;
    if (fmt.fmt.pix.sizeimage < min)
        fmt.fmt.pix.sizeimage = min;

    init_userp(fmt.fmt.pix.sizeimage);

    const int error = nvCreateAprilTagsDetector(
        &april_tags_handle, width, height, tile_size, cuAprilTagsFamily::NVAT_TAG36H11,
        //&april_tags_handle, width, height, cuAprilTagsFamily::NVAT_TAG16H5,
        &cam_intrinsics, tag_size);
    if (error != 0) {
        throw std::runtime_error(
            "Failed to create NV April Tags detector (error code " +
            std::to_string(error) + ")");
    }
    // Create stream for detection
    cudaStreamCreate(&main_stream);

    // Allocate the output vector to contain detected AprilTags.
    tags.resize(max_tags);
    input_image.width = width;
    input_image.height = height;
    input_image.pitch = 3;  // not sure
    // set input_image.dev_ptr to buffer before detecting
}

static void
close_device(void) {
    if(gpu_detector_) free(gpu_detector_);
    if (-1 == close(fd))
        errno_exit("close");

    fd = -1;
}

static void
open_device(void) {
    struct stat st;

    if (-1 == stat(dev_name, &st)) {
        fprintf(stderr, "Cannot identify '%s': %d, %s\n",
                dev_name, errno, strerror(errno));
        exit(EXIT_FAILURE);
    }

    if (!S_ISCHR(st.st_mode)) {
        fprintf(stderr, "%s is no device\n", dev_name);
        exit(EXIT_FAILURE);
    }

    fd = open(dev_name, O_RDWR /* required */ | O_NONBLOCK, 0);

    if (-1 == fd) {
        fprintf(stderr, "Cannot open '%s': %d, %s\n",
                dev_name, errno, strerror(errno));
        exit(EXIT_FAILURE);
    }
}

static void
init_cuda(void) {
    // 971 detector
    gpu_detector_ = new frc971::apriltag::GpuDetector(width, height, MakeTagDetector(tag36h11_create()), create_camera_matrix(),
                      create_distortion_coefficients());

    /* Check unified memory support. */
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);
    if (!devProp.managedMemory) {
        printf("CUDA device does not support managed memory.\n");
    }

    /* Allocate output buffer. */
    size_t size = width * height * 3 * sizeof(char);
    cudaMallocManaged(&cuda_out_buffer, size, cudaMemAttachGlobal);

    cudaDeviceSynchronize();
}

void print_pose3d(frc::Pose3d pos, std::string description) {
        std::cout << description << " " << pos.X().value() << " " << pos.Y().value() << " " << pos.Z().value() << std::endl;
        std::cout << "   rotation " << pos.Rotation().X().value() << " " << pos.Rotation().Y().value() << " " << pos.Rotation().Z().value() << std::endl;
};


frc::Pose3d get_field_tag_corner(frc::Pose3d pos, int corner) {
	switch(corner) {
	   case 0:
		return pos + frc::Transform3d(frc::Translation3d(0_m,-HALFTAGSIZE,HALFTAGSIZE),frc::Rotation3d());
	   case 1:
		return pos + frc::Transform3d(frc::Translation3d(0_m,HALFTAGSIZE,HALFTAGSIZE),frc::Rotation3d());
	   case 2:
		return pos + frc::Transform3d(frc::Translation3d(0_m,HALFTAGSIZE,-HALFTAGSIZE),frc::Rotation3d());
	   case 3:
		return pos + frc::Transform3d(frc::Translation3d(0_m,-HALFTAGSIZE,-HALFTAGSIZE),frc::Rotation3d());
	   default:
		return frc::Pose3d();
	}
}

static void
init_wpilib(void) {
    std::cout << "Setting up NetworkTables client for team " << team << std::endl;
    ntinst.StartClient4("ov9281");
    ntinst.SetServerTeam(team);

    auto table = ntinst.GetTable("CameraPublisher");
#if DEVICE == 1
    std::string mjpgstream[] = {"http://10.41.43.48:1183/stream.mjpg"};
#else
    std::string mjpgstream[] = {"http://10.41.43.48:1181/stream.mjpg"};
#endif
    table->PutStringArray("Jetson/streams", std::span{mjpgstream});

    cvMjpegServer.SetSource(cvsource);
    CS_Status status = 0;
    cs::AddListener(
        [&](const cs::RawEvent &event) {
            fmt::print("FPS={} MBPS={}\n", cvsource.GetActualFPS(),
                       (cvsource.GetActualDataRate() / 1000000.0));
        },
        cs::RawEvent::kTelemetryUpdated, false, &status);
    cs::SetTelemetryPeriod(1.0);
    cvsource.CreateProperty("gain", cs::VideoProperty::kInteger, OV9281_MIN_GAIN, OV9281_MAX_GAIN, 1, OV9281_DEFAULT_GAIN, OV9281_DEFAULT_GAIN);
    cvsource.CreateProperty("exposure", cs::VideoProperty::kInteger, OV9281_MIN_EXPOSURE_COARSE, OV9281_MAX_EXPOSURE_COARSE, 1, OV9281_DEFAULT_EXPOSURE_COARSE, OV9281_DEFAULT_EXPOSURE_COARSE);
    cs::AddListener(
        [&](const cs::RawEvent &event) {
            fmt::print("{}={}\n", event.name, event.value);
            if (event.name.compare("gain") == 0)
                setgain(event.value);
            if (event.name.compare("exposure") == 0)
                setexposure(event.value);
        },
        cs::RawEvent::kSourcePropertyValueUpdated, false, &status);

    cvMjpegServer2.SetSource(cvsource2);

    fieldMatEmpty = cv::Mat::zeros(cv::Size(fieldWidth, fieldHeight), CV_8UC3);
    fieldlayout = frc::LoadAprilTagLayoutField(frc::AprilTagField::k2024Crescendo);
    drawField(fieldMatEmpty);  // just draw field once
    std::cout << "Field tag positions" << std::endl;
    for (int i = 1; i < 586; i++) {
	if(!fieldlayout.GetTagPose(i)) break;
        auto pos = fieldlayout.GetTagPose(i).value_or(frc::Pose3d());
	print_pose3d(pos, std::to_string(i));
        auto corner0 = get_field_tag_corner(pos, 0);
        auto corner1 = get_field_tag_corner(pos, 1);
        auto corner2 = get_field_tag_corner(pos, 2);
        auto corner3 = get_field_tag_corner(pos, 3);
	print_pose3d(corner0, "corner0");
	print_pose3d(corner1, "corner1");
	print_pose3d(corner2, "corner2");
	print_pose3d(corner3, "corner3");
    }
    intrinsicMat.at<double>(0, 0) = cam_intrinsics.fx;
    intrinsicMat.at<double>(1, 0) = 0;
    intrinsicMat.at<double>(2, 0) = 0;

    intrinsicMat.at<double>(0, 1) = 0;
    intrinsicMat.at<double>(1, 1) = cam_intrinsics.fy;
    intrinsicMat.at<double>(2, 1) = 0;

    intrinsicMat.at<double>(0, 2) = cam_intrinsics.cx;
    intrinsicMat.at<double>(1, 2) = cam_intrinsics.cy;
    intrinsicMat.at<double>(2, 2) = 1;
}

static void
usage(FILE *fp,
      int argc,
      char **argv) {
    fprintf(fp,
            "Usage: %s [options]\n\n"
            "Options:\n"
            "-d | --device name   Video device name (default: %s)\n"
            "-h | --help          Print this message\n"
            "",
            argv[0], dev_name);
}

static const char short_options[] = "d:hmo";

static const struct option
    long_options[] = {
        {"device", required_argument, NULL, 'd'},
        {"help", no_argument, NULL, 'h'},
        {0, 0, 0, 0}};

int main(int argc,
         char **argv) {
    for (;;) {
        int index;
        int c;

        c = getopt_long(argc, argv,
                        short_options, long_options,
                        &index);

        if (-1 == c)
            break;

        switch (c) {
            case 0: /* getopt_long() flag */
                break;

            case 'd':
                dev_name = optarg;
                break;

            case 'h':
                usage(stdout, argc, argv);
                exit(EXIT_SUCCESS);

            default:
                usage(stderr, argc, argv);
                exit(EXIT_FAILURE);
        }
    }

    init_wpilib();

    open_device();

    init_device();

    init_cuda();

    start_capturing();

    mainloop();

    stop_capturing();

    uninit_device();

    close_device();

    exit(EXIT_SUCCESS);

    return 0;
}
