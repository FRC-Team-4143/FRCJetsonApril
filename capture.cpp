/*
 * Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

/*
 *  V4L2 video capture example
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include <getopt.h>             /* getopt_long() */

#include <fcntl.h>              /* low-level i/o */
#include <unistd.h>
#include <errno.h>
#include <malloc.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <chrono>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <fmt/format.h>

#include <asm/types.h>          /* for videodev2.h */

#include <iostream>
#include <time.h>
#include <linux/videodev2.h>

#include <cuda_runtime.h>

#include <cscore.h>
#include <cscore_cv.h>
#include <networktables/NetworkTableInstance.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/cudafilters.hpp>

#include "cuAprilTags.h"
#include <eigen3/Eigen/Dense>

#include "raw2rgb.cuh"

#define OV9281_MIN_GAIN                 0x0000
#define OV9281_MAX_GAIN                 0x00FE
#define OV9281_DEFAULT_GAIN             0x0010 /* 1.0x real gain */

#define OV9281_MIN_EXPOSURE_COARSE      0x00000010
#define OV9281_MAX_EXPOSURE_COARSE      0x00003750
#define OV9281_DEFAULT_EXPOSURE_COARSE  0x00002A90

#define MAX_TAG_ID 8

#define CLEAR(x) memset (&(x), 0, sizeof (x))
#define ARRAY_SIZE(a)   (sizeof(a)/sizeof((a)[0]))

#define EIGHTBIT  // otherwise 10bit input

struct buffer {
    void *                  start;
    size_t                  length;
};

const int team = 4143;

// Handle used to interface with the stereo library.
cuAprilTagsHandle april_tags_handle = nullptr;

// Camera intrinsics
// Innovision OV9281rawv2
cuAprilTagsCameraIntrinsics_t cam_intrinsics = 
   {1082., 1082., 1280.0/2.0, 800.0/2.0};

cv::Mat intrinsicMat(3, 3, cv::DataType<double>::type);
std::vector<cv::Point3d> objectPoints;
//cv::Mat distCoeffs = (cv::Mat_<double>(5, 1) << -0.4647801828154495, 0.2844034418347612, -0.008834734456932225, -0.01218903939069973, -0.1319188405435461);
cv::Mat distCoeffs = (cv::Mat_<double>(5, 1) << 0.0, 0.0, 0.0, 0.0, 0.0);

// Output vector of detected Tags
std::vector<cuAprilTagsID_t> tags;

// CUDA stream
cudaStream_t main_stream = {};

float tag_size = .16;  //same units as camera calib
int max_tags = 5;
int tile_size = 24;

cuAprilTagsImageInput_t input_image;


static const char *     dev_name        = "/dev/video0";
static int              fd              = -1;
struct buffer *         buffers         = NULL;
static unsigned int     n_buffers       = 0;
static unsigned int     width           = 1280;
static unsigned int     height          = 720;
static unsigned int     count           = 0;
static unsigned char *  cuda_out_buffer = NULL;
#ifdef EIGHTBIT
static unsigned int     pixel_format    = V4L2_PIX_FMT_SRGGB8;
#else
static unsigned int     pixel_format    = V4L2_PIX_FMT_SRGGB10;
#endif
static unsigned int     field           = V4L2_FIELD_NONE;

auto ntinst = nt::NetworkTableInstance::GetDefault();

cs::CvSource cvsource{"cvsource", cs::VideoMode::kMJPEG, (int) width, (int) height, 30};
cs::MjpegServer cvMjpegServer{"cvhttpserver", 1181};

const std::string_view config = "{ }";

static void
errno_exit                      (const char *           s)
{
    fprintf (stderr, "%s error %d, %s\n",
            s, errno, strerror (errno));

    exit (EXIT_FAILURE);
}

static int
xioctl                          (int                    fd,
                                 int                    request,
                                 void *                 arg)
{
    int r;

    do r = ioctl (fd, request, arg);
    while (-1 == r && EINTR == errno);

    return r;
}

static void
process_image (void *           p, double fps)
{
#ifdef EIGHTBIT
    gpuConvertgraytoRGB ((unsigned char *) p, cuda_out_buffer, width, height, main_stream);
#else
    gpuConvertgraytoRGB ((unsigned short *) p, cuda_out_buffer, width, height, main_stream);
#endif

    uint32_t num_detections;
    input_image.dev_ptr = (uchar3*)cuda_out_buffer;
    input_image.pitch = width*3;
    //cudaStreamAttachMemAsync(main_stream, input_image.dev_ptr, 0, cudaMemAttachGlobal);
    const int error = cuAprilTagsDetect(
      april_tags_handle, &input_image, tags.data(),
      &num_detections, max_tags, main_stream);
    //cudaStreamAttachMemAsync(main_stream, input_image.dev_ptr, 0, cudaMemAttachHost);
    //cudaStreamSynchronize(main_stream);

    if(error != 0) {
	    std::cout << "april tag detect error" << std::endl;
    }
    
    //if (num_detections > 0) {
    //	std::cout << "frame " << count << " found tag";
    //}

    for (uint32_t i = 0; i < num_detections; i++) {
       const cuAprilTagsID_t & detection = tags[i];
       if(detection.id > MAX_TAG_ID) continue;

       float distance = std::sqrt(
		       detection.translation[0] * detection.translation[0] +
		       detection.translation[1] * detection.translation[1] +
		       detection.translation[2] * detection.translation[2]);

       std::cout << "tag" <<  detection.id << " error: " << unsigned(detection.hamming_error) << " " << detection.translation[0];
       std::cout << "," << detection.translation[1];
       std::cout << "," << detection.translation[2] << " " << distance;

       //const Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::ColMajor>>
       //     orientation(detection.orientation);
       //std::cout << std::endl << orientation << std::endl;
       //const Eigen::Quaternion<float> q(orientation);
       //std::cout << "quaternion: " << q.w() << " " << q.x() << " " << q.y() << " " << q.z() << std::endl;
       //const Eigen::AngleAxis<float> axis(q);
       //std::cout << "angle: " << axis.angle()*180.0/M_PI << std::endl;
       //
       //double zsqr = q.z() * q.z();
       //double t0 = -2.0 * (zsqr + q.w() * q.w()) + 1.0;
       //double t1 = +2.0 * (q.y() * q.z() + q.x() * q.w());
       //double t2 = -2.0 * (q.y() * q.w() - q.x() * q.z());
       //double t3 = +2.0 * (q.z() * q.w() + q.x() * q.y());
       //double t4 = -2.0 * (q.y() * q.y() + zsqr) + 1.0;

       //t2 = t2 > 1.0 ? 1.0 : t2;
       //t2 = t2 < -1.0 ? -1.0 : t2;

       //double roll = atan2(t3, t4)*180/M_PI;
       //double yaw = asin(t2)*180/M_PI;
       //double pitch = atan2(t1, t0)*180/M_PI;
       //std::cout << "roll: " << roll << " pitch: " << pitch << " yaw: " << yaw << std::endl;


    }
    if (num_detections > 0)
    	std::cout << std::endl << std::endl ;

    if(count % 8 == 0) {
        cv::Mat mat(height, width, CV_8UC3, cuda_out_buffer);

	    std::vector<cv::Point2d> imagePoints;
	    cv::Mat rRot(3, 3, cv::DataType<double>::type, 0.0);
	    cv::Mat rVec(3, 1, cv::DataType<double>::type, 0.0);
	    cv::Mat tVec(3, 1, cv::DataType<double>::type, 0.0);

    	for (uint32_t i = 0; i < num_detections; i++) {
       		const cuAprilTagsID_t & detection = tags[i];
                if(detection.id > MAX_TAG_ID) continue;

       		cv::Point tag_points[4];
	       tag_points[0] = cv::Point( detection.corners[0].x, detection.corners[0].y);
	       tag_points[1] = cv::Point( detection.corners[1].x, detection.corners[1].y);
	       tag_points[2] = cv::Point( detection.corners[2].x, detection.corners[2].y);
	       tag_points[3] = cv::Point( detection.corners[3].x, detection.corners[3].y);
	       cv::line(mat, tag_points[0], tag_points[1], cv::Scalar(255,0,0), 2);
	       cv::line(mat, tag_points[1], tag_points[2], cv::Scalar(255,0,0), 2);
	       cv::line(mat, tag_points[2], tag_points[3], cv::Scalar(255,0,0), 2);
	       cv::line(mat, tag_points[3], tag_points[0], cv::Scalar(255,0,0), 2);

	       for(int i = 0; i < 9; i++) rRot.at<double>(i) = detection.orientation[i];
	       cv::transpose(rRot,rRot);
	       cv::Rodrigues(rRot, rVec);

	       tVec.at<double>(0) = detection.translation[0];
	       tVec.at<double>(1) = detection.translation[1];
	       tVec.at<double>(2) = detection.translation[2];

	       //std::cout << "Intrisic matrix: " << intrinsicMat << std::endl << std::endl;
	       //std::cout << "Rotation matrix: " << rRot << std::endl << std::endl;
	       //std::cout << "Rotation vector: " << rVec << std::endl << std::endl;
	       //std::cout << "Translation vector: " << tVec << std::endl << std::endl;
	       cv::projectPoints(objectPoints, rVec, tVec, intrinsicMat, distCoeffs, imagePoints);
		//for (unsigned int i = 0; i < imagePoints.size(); ++i){
		//  cv::circle(mat, imagePoints[i], 3, cv::Scalar(0,255,0), 1);
		//}
	        cv::line(mat, imagePoints[0], imagePoints[1], cv::Scalar(0,255,0), 2);
	        cv::line(mat, imagePoints[1], imagePoints[2], cv::Scalar(0,255,0), 2);
	        cv::line(mat, imagePoints[2], imagePoints[3], cv::Scalar(0,255,0), 2);
	        cv::line(mat, imagePoints[3], imagePoints[0], cv::Scalar(0,255,0), 2);
	        cv::line(mat, imagePoints[4], imagePoints[5], cv::Scalar(0,255,0), 2);
	        cv::line(mat, imagePoints[5], imagePoints[6], cv::Scalar(0,255,0), 2);
	        cv::line(mat, imagePoints[6], imagePoints[7], cv::Scalar(0,255,0), 2);
	        cv::line(mat, imagePoints[7], imagePoints[4], cv::Scalar(0,255,0), 2);
	        cv::line(mat, imagePoints[0], imagePoints[4], cv::Scalar(0,255,0), 2);
	        cv::line(mat, imagePoints[1], imagePoints[5], cv::Scalar(0,255,0), 2);
	        cv::line(mat, imagePoints[2], imagePoints[6], cv::Scalar(0,255,0), 2);
	        cv::line(mat, imagePoints[3], imagePoints[7], cv::Scalar(0,255,0), 2);
	        cv::line(mat, imagePoints[8], imagePoints[9], cv::Scalar(255,0,0), 2);
	        cv::line(mat, imagePoints[8], imagePoints[10], cv::Scalar(0,255,0), 2);
	        cv::line(mat, imagePoints[8], imagePoints[11], cv::Scalar(0,0,255), 2);
        	std::string str = std::to_string(detection.id);
        	cv::putText(mat, str, imagePoints[8],cv::FONT_HERSHEY_DUPLEX,1,cv::Scalar(0,255,255),2,false);

	}

        std::string str = "fps: " + std::to_string(fps);
        cv::putText(mat, str, cv::Point(50,50),cv::FONT_HERSHEY_DUPLEX,1,cv::Scalar(255,0,0),2,false);
	cvsource.PutFrame(mat);
    }
}

static int
read_frame                      (double fps)
{
    struct v4l2_buffer buf;
    unsigned int i;

    CLEAR (buf);

    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_USERPTR;

    auto start = std::chrono::high_resolution_clock::now();
    if (-1 == xioctl (fd, VIDIOC_DQBUF, &buf)) {
	switch (errno) {
	    case EAGAIN:
		return 0;

	    case EIO:
		/* Could ignore EIO, see spec. */

		/* fall through */

	    default:
		errno_exit ("VIDIOC_DQBUF");
	}
    }

    for (i = 0; i < n_buffers; ++i)
	if (buf.m.userptr == (unsigned long) buffers[i].start
		&& buf.length == buffers[i].length)
	    break;

    assert (i < n_buffers);

    fd_set fds;
    struct timeval tv;
    int r;

    FD_ZERO (&fds);
    FD_SET (fd, &fds);

    /* Timeout. */
    tv.tv_sec = 0;
    tv.tv_usec = 0;

    r = select (fd + 1, &fds, NULL, NULL, &tv);

    if(r == 0) {
       //std::cout << "buffer timestamp " << buf.timestamp.tv_sec << " " << buf.timestamp.tv_usec << std::endl;
       process_image ((void *) buf.m.userptr, fps);
    }
    else {
	//std::cout << "another frame available skipping this one" << std::endl;
    }


    if (-1 == xioctl (fd, VIDIOC_QBUF, &buf))
	errno_exit ("VIDIOC_QBUF");

    if(r == 0){ 
    	//auto end = std::chrono::high_resolution_clock::now();
	//double milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	//std::cout << "latency (ms): " << milliseconds << std::endl;
        return 1;
    }
    else return 0;
}

static void
mainloop                        (void)
{
    double fps  = 0;

    while(true) {
	    count = 0;
	    auto start = std::chrono::high_resolution_clock::now();
	    while (count < 120) {
		for (;;) {
		    fd_set fds;
		    struct timeval tv;
		    int r;

		    FD_ZERO (&fds);
		    FD_SET (fd, &fds);

		    /* Timeout. */
		    tv.tv_sec = 1;
		    tv.tv_usec = 0;

		    r = select (fd + 1, &fds, NULL, NULL, &tv);

		    if (-1 == r) {
			if (EINTR == errno)
			    continue;

			errno_exit ("select");
		    }

		    if (0 == r) {
			fprintf (stderr, "select timeout\n");
			exit (EXIT_FAILURE);
		    }

		    if (read_frame (fps)) {
		        count++;
			break;
		    }

		    /* EAGAIN - continue select loop. */
		}
	    }

	    auto end = std::chrono::high_resolution_clock::now();
	    double seconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0;
	    fps  = ((double)count) / seconds;

            std::string str = "fps: " + std::to_string(fps) + " " + std::to_string(count) + "/" + std::to_string(seconds);
	    std::cout << str << std::endl;
    }

}

static void
stop_capturing                  (void)
{
    enum v4l2_buf_type type;

    type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

    if (-1 == xioctl (fd, VIDIOC_STREAMOFF, &type))
	errno_exit ("VIDIOC_STREAMOFF");
}

static void
start_capturing                 (void)
{
    unsigned int i;
    enum v4l2_buf_type type;

    for (i = 0; i < n_buffers; ++i) {
	struct v4l2_buffer buf;

	CLEAR (buf);

	buf.type        = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	buf.memory      = V4L2_MEMORY_USERPTR;
	buf.index       = i;
	buf.m.userptr   = (unsigned long) buffers[i].start;
	buf.length      = buffers[i].length;

	if (-1 == xioctl (fd, VIDIOC_QBUF, &buf))
	    errno_exit ("VIDIOC_QBUF");
    }

    type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

    if (-1 == xioctl (fd, VIDIOC_STREAMON, &type))
	errno_exit ("VIDIOC_STREAMON");

}

static void
uninit_device                   (void)
{
    unsigned int i;

    for (i = 0; i < n_buffers; ++i) {
	cudaFree (buffers[i].start);
    }

    free (buffers);

    cudaFree (cuda_out_buffer);
}

static void
init_userp                      (unsigned int           buffer_size)
{
    struct v4l2_requestbuffers req;
    unsigned int page_size;

    page_size = getpagesize ();
    buffer_size = (buffer_size + page_size - 1) & ~(page_size - 1);

    CLEAR (req);

    req.count               = 4;
    req.type                = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory              = V4L2_MEMORY_USERPTR;

    if (-1 == xioctl (fd, VIDIOC_REQBUFS, &req)) {
        if (EINVAL == errno) {
            fprintf (stderr, "%s does not support "
                    "user pointer i/o\n", dev_name);
            exit (EXIT_FAILURE);
        } else {
            errno_exit ("VIDIOC_REQBUFS");
        }
    }

    buffers = (struct buffer *) calloc (4, sizeof (*buffers));

    if (!buffers) {
        fprintf (stderr, "Out of memory\n");
        exit (EXIT_FAILURE);
    }

    for (n_buffers = 0; n_buffers < 4; ++n_buffers) {
        buffers[n_buffers].length = buffer_size;
        cudaMallocManaged (&buffers[n_buffers].start, buffer_size, cudaMemAttachGlobal);

        if (!buffers[n_buffers].start) {
            fprintf (stderr, "Out of memory\n");
            exit (EXIT_FAILURE);
        }
    }
}

static void
setgain                     (int gain)
{
    struct v4l2_control control;
    struct v4l2_ext_controls ext_controls;
    struct v4l2_ext_control ext_control;

    CLEAR(ext_controls);
    CLEAR(ext_control);
    ext_controls.ctrl_class = V4L2_CTRL_ID2CLASS(0x009a2000);
    ext_controls.count = 1;
    ext_controls.controls = &ext_control;

    ext_control.id = 0x009a2009; // gain
    ext_control.value64 = gain; 
    if (-1 == xioctl (fd, VIDIOC_S_EXT_CTRLS, &ext_controls)) {
       errno_exit ("VIDIOC_S_CTRL gain");
    }
}

static void
setexposure                   (int exposure)
{
    struct v4l2_control control;
    struct v4l2_ext_controls ext_controls;
    struct v4l2_ext_control ext_control;

    CLEAR(ext_controls);
    CLEAR(ext_control);
    ext_controls.ctrl_class = V4L2_CTRL_ID2CLASS(0x009a2000);
    ext_controls.count = 1;
    ext_controls.controls = &ext_control;

    ext_control.id = 0x009a2001; // coarse exposure
    ext_control.value64 = exposure; 
    if (-1 == xioctl (fd, VIDIOC_S_EXT_CTRLS, &ext_controls)) {
        errno_exit ("VIDIOC_S_CTRL exposure");
    }
}

static void
init_device                     (void)
{
    struct v4l2_capability cap;
    struct v4l2_cropcap cropcap;
    struct v4l2_crop crop;
    struct v4l2_format fmt;
    struct v4l2_control control;
    struct v4l2_ext_controls ext_controls;
    struct v4l2_ext_control ext_control;
    unsigned int min;

    if (-1 == xioctl (fd, VIDIOC_QUERYCAP, &cap)) {
        if (EINVAL == errno) {
            fprintf (stderr, "%s is no V4L2 device\n",
                    dev_name);
            exit (EXIT_FAILURE);
        } else {
            errno_exit ("VIDIOC_QUERYCAP");
        }
    }

    if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
        fprintf (stderr, "%s is no video capture device\n",
                dev_name);
        exit (EXIT_FAILURE);
    }

    if (!(cap.capabilities & V4L2_CAP_STREAMING)) {
	fprintf (stderr, "%s does not support streaming i/o\n",
		dev_name);
	exit (EXIT_FAILURE);
    }


    CLEAR(control);
    control.id = 0x009a2064; // BYPASS_MODE
    control.value = 0;
    if (-1 == xioctl (fd, VIDIOC_S_CTRL, &control)) {
        errno_exit ("VIDIOC_S_CTRL bypass mode");
    }

    CLEAR(control);
    control.id = 0x009a206d; // low_latency_mode
    control.value = 0;
    if (-1 == xioctl (fd, VIDIOC_S_CTRL, &control)) {
        errno_exit ("VIDIOC_S_CTRL low latency mode");
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

    ext_control.id = 0x009a2009; // gain
    ext_control.value64 = OV9281_DEFAULT_GAIN; 
    if (-1 == xioctl (fd, VIDIOC_S_EXT_CTRLS, &ext_controls)) {
       errno_exit ("VIDIOC_S_CTRL gain");
    }

    //ext_control.id = 0x009a200a; // exposure
    //ext_control.value64 = 0x10; 
    //ext_control.value64 = 0; 
    //if (-1 == xioctl (fd, VIDIOC_S_EXT_CTRLS, &ext_controls)) {
    //    errno_exit ("VIDIOC_S_CTRL exposure");
    //}

    ext_control.id = 0x009a2001; // coarse exposure
    ext_control.value64 = OV9281_DEFAULT_EXPOSURE_COARSE; 
    if (-1 == xioctl (fd, VIDIOC_S_EXT_CTRLS, &ext_controls)) {
        errno_exit ("VIDIOC_S_CTRL exposure");
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


    CLEAR (fmt);

    fmt.type                = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width       = width;
    fmt.fmt.pix.height      = height;
    fmt.fmt.pix.pixelformat = pixel_format;
    fmt.fmt.pix.field       = field;

    if (-1 == xioctl (fd, VIDIOC_S_FMT, &fmt))
        errno_exit ("VIDIOC_S_FMT");

    /* Note VIDIOC_S_FMT may change width and height. */
    fprintf (stderr, "width %d height %d bytesperline %d sizeimage %d\n",
                fmt.fmt.pix.width, fmt.fmt.pix.height, 
		fmt.fmt.pix.bytesperline, fmt.fmt.pix.sizeimage);

    /* Buggy driver paranoia. */
    min = fmt.fmt.pix.width * 2;
    if (fmt.fmt.pix.bytesperline < min)
        fmt.fmt.pix.bytesperline = min;
    min = fmt.fmt.pix.bytesperline * fmt.fmt.pix.height;
    if (fmt.fmt.pix.sizeimage < min)
        fmt.fmt.pix.sizeimage = min;

    init_userp (fmt.fmt.pix.sizeimage);

    const int error = nvCreateAprilTagsDetector(
      &april_tags_handle, width, height, tile_size, cuAprilTagsFamily::NVAT_TAG16H5,
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
    input_image.pitch = 3; // not sure
    // set input_image.dev_ptr to buffer before detecting
}

static void
close_device                    (void)
{
    if (-1 == close (fd))
        errno_exit ("close");

    fd = -1;
}

static void
open_device                     (void)
{
    struct stat st;

    if (-1 == stat (dev_name, &st)) {
        fprintf (stderr, "Cannot identify '%s': %d, %s\n",
                dev_name, errno, strerror (errno));
        exit (EXIT_FAILURE);
    }

    if (!S_ISCHR (st.st_mode)) {
        fprintf (stderr, "%s is no device\n", dev_name);
        exit (EXIT_FAILURE);
    }

    fd = open (dev_name, O_RDWR /* required */ | O_NONBLOCK, 0);

    if (-1 == fd) {
        fprintf (stderr, "Cannot open '%s': %d, %s\n",
                dev_name, errno, strerror (errno));
        exit (EXIT_FAILURE);
    }
}

static void
init_cuda                       (void)
{
    /* Check unified memory support. */
    cudaDeviceProp devProp;
    cudaGetDeviceProperties (&devProp, 0);
    if (!devProp.managedMemory) {
        printf ("CUDA device does not support managed memory.\n");
    }

    /* Allocate output buffer. */
    size_t size = width * height * 3 * sizeof(char);
    cudaMallocManaged (&cuda_out_buffer, size, cudaMemAttachGlobal);

    cudaDeviceSynchronize ();
}

static void
init_wpilib                       (void)
{
    std::cout << "Setting up NetworkTables client for team " << team << std::endl;
    ntinst.StartClient4("ov9281");
    ntinst.SetServerTeam(team);
    auto table = ntinst.GetTable("CameraPublisher");
    std::string mjpgstream[] = {"http://10.41.43.48:1181/stream.mjpg"};
    table->PutStringArray("Jetson/streams", std::span{mjpgstream});

    cvMjpegServer.SetSource(cvsource);
    CS_Status status = 0;
    cs::AddListener(
      [&](const cs::RawEvent& event) {
        fmt::print("FPS={} MBPS={}\n", cvsource.GetActualFPS(),
                   (cvsource.GetActualDataRate() / 1000000.0));
      },
    cs::RawEvent::kTelemetryUpdated, false, &status);
    cs::SetTelemetryPeriod(1.0);
    cvsource.CreateProperty("gain", cs::VideoProperty::kInteger, OV9281_MIN_GAIN, OV9281_MAX_GAIN, 1, OV9281_DEFAULT_GAIN, OV9281_DEFAULT_GAIN);
    cvsource.CreateProperty("exposure", cs::VideoProperty::kInteger, OV9281_MIN_EXPOSURE_COARSE, OV9281_MAX_EXPOSURE_COARSE, 1, OV9281_DEFAULT_EXPOSURE_COARSE, OV9281_DEFAULT_EXPOSURE_COARSE);
    cs::AddListener(
      [&](const cs::RawEvent& event) {
        fmt::print("{}={}\n", event.name, event.value);
	if(event.name.compare("gain") == 0)
	   setgain(event.value);
	if(event.name.compare("exposure") == 0)
	   setexposure(event.value);
      },
    cs::RawEvent::kSourcePropertyValueUpdated, false, &status);


    intrinsicMat.at<double>(0, 0) = cam_intrinsics.fx;
    intrinsicMat.at<double>(1, 0) = 0;
    intrinsicMat.at<double>(2, 0) = 0;

    intrinsicMat.at<double>(0, 1) = 0;
    intrinsicMat.at<double>(1, 1) = cam_intrinsics.fy;
    intrinsicMat.at<double>(2, 1) = 0;

    intrinsicMat.at<double>(0, 2) = cam_intrinsics.cx;
    intrinsicMat.at<double>(1, 2) = cam_intrinsics.cy;
    intrinsicMat.at<double>(2, 2) = 1;
    objectPoints.push_back(cv::Point3d(-.1,.1,0));
    objectPoints.push_back(cv::Point3d(-.1,-.1,0));
    objectPoints.push_back(cv::Point3d(.1,-.1,0));
    objectPoints.push_back(cv::Point3d(.1,.1,0));
    objectPoints.push_back(cv::Point3d(-.1,.1,-.2));
    objectPoints.push_back(cv::Point3d(-.1,-.1,-.2));
    objectPoints.push_back(cv::Point3d(.1,-.1,-.2));
    objectPoints.push_back(cv::Point3d(.1,.1,-.2));
    objectPoints.push_back(cv::Point3d(0,0,0));
    objectPoints.push_back(cv::Point3d(.1,0,0));
    objectPoints.push_back(cv::Point3d(0,.1,0));
    objectPoints.push_back(cv::Point3d(0,0,-.1));
}

static void
usage                           (FILE *                 fp,
                                 int                    argc,
                                 char **                argv)
{
    fprintf (fp,
            "Usage: %s [options]\n\n"
            "Options:\n"
            "-d | --device name   Video device name (default: %s)\n"
            "-h | --help          Print this message\n"
            "",
            argv[0], dev_name);
}

static const char short_options [] = "d:hmo";

static const struct option
long_options [] = {
    { "device",     required_argument,      NULL,           'd' },
    { "help",       no_argument,            NULL,           'h' },
    { 0, 0, 0, 0 }
};

int
main                            (int                    argc,
                                 char **                argv)
{
    for (;;) {
        int index;
        int c;

        c = getopt_long (argc, argv,
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
                usage (stdout, argc, argv);
                exit (EXIT_SUCCESS);

            default:
                usage (stderr, argc, argv);
                exit (EXIT_FAILURE);
        }
    }

    init_wpilib ();

    open_device ();

    init_device ();

    init_cuda ();

    start_capturing ();

    mainloop ();

    stop_capturing ();

    uninit_device ();

    close_device ();

    exit (EXIT_SUCCESS);

    return 0;
}
