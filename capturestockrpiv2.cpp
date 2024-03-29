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
#include <sys/time.h>
#include <sys/mman.h>
#include <sys/ioctl.h>

#include <asm/types.h>          /* for videodev2.h */

#include <iostream>
#include <time.h>
#include <linux/videodev2.h>

#include <npp.h>
#include <cuda_runtime.h>
#include "helper_cuda.h"

#define opencv
#ifdef opencv
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/core/utility.hpp>

#define APRILTAGS
#ifdef APRILTAGS
#include "cuAprilTags.h"
#include <eigen3/Eigen/Dense>
#endif


//#define CALIB
#ifdef CALIB
#include <opencv2/calib3d.hpp>
std::vector< std::vector< cv::Point3f > > object_points;
std::vector< std::vector< cv::Point2f > > image_points;
std::vector< cv::Point2f > corners;

int board_width = 7;
int board_height = 5;

cv::Size board_size = cv::Size(board_width, board_height);
int board_n = board_width * board_height;
float square_size = 0.028;  // in meters?
#endif
#endif

#include "raw2rgb.cuh"

#define CLEAR(x) memset (&(x), 0, sizeof (x))
#define ARRAY_SIZE(a)   (sizeof(a)/sizeof((a)[0]))

struct buffer {
    void *                  start;
    size_t                  length;
};

#ifdef APRILTAGS
// Handle used to interface with the stereo library.
cuAprilTagsHandle april_tags_handle = nullptr;

// Camera intrinsics
cuAprilTagsCameraIntrinsics_t cam_intrinsics = 
   {575.1886716994702, 581.8990918151019, 296.0331275208252, 217.670344390834};

// Output vector of detected Tags
std::vector<cuAprilTagsID_t> tags;

// CUDA stream
cudaStream_t main_stream = {};

float tag_size = .16;  //same units as camera calib
int max_tags = 5;
int tile_size = 24;

cuAprilTagsImageInput_t input_image;

// Size of image buffer
size_t input_image_buffer_size = 0;

#endif


static const char *     dev_name        = "/dev/video0";
static int              fd              = -1;
struct buffer *         buffers         = NULL;
static unsigned int     n_buffers       = 0;
static unsigned int     width           = 1280; // mode 4 or 5
static unsigned int     height          = 720;
//static unsigned int     width           = 1920; // mode 2
//static unsigned int     height          = 1080;
//static unsigned int     width           = 1640; // mode 3
//static unsigned int     height          = 1232;
static unsigned int     count           = 0;
static unsigned char *  cuda_out_buffer = NULL;
static const char *     file_name       = "out.ppm";
static unsigned int     pixel_format    = V4L2_PIX_FMT_SRGGB10;
//static unsigned int     pixel_format    = V4L2_PIX_FMT_SRGGB8;
static unsigned int     field           = V4L2_FIELD_NONE;


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
    NppiSize osize;
    osize.width = width;
    osize.height = height;
    NppiRect orect;
    orect.x = 0;
    orect.y = 0;
    orect.width = width;
    orect.height = height;

    NppStatus eStatusNPP; 
    //gpuConvertrawtoRGB ((unsigned short *) p, cuda_out_buffer, width, height);
    //gpuConvertrawtoRGB ((unsigned char *) p, cuda_out_buffer, width, height);
    //gpuConvertrawtoRGBA ((unsigned char *) p, cuda_out_buffer, width, height);
    //gpuConvertgraytoRGBA ((unsigned char *) p, cuda_out_buffer, width, height);
    //gpuConvertgraytoRGB ((unsigned short *) p, cuda_out_buffer, width, height);
    eStatusNPP = nppiCFAToRGB_16u_C1C3R((Npp16u*) p, width*sizeof(short), osize, orect, (Npp16u*) cuda_out_buffer, width*3*sizeof(short), NPPI_BAYER_RGGB, NPPI_INTER_UNDEFINED);
    if (eStatusNPP != NPP_SUCCESS) std::cout << "NPP_CHECK_NPP - eStatusNPP = " << _cudaGetErrorEnum(eStatusNPP) << "("<< eStatusNPP << ")" << std::endl;

    /* Save image. */
    /*
    if (count == 0) {
        printf ("CUDA format conversion on frame %p %d %d\n", p, width, height);
        FILE *fp = fopen (file_name, "wb");
        fprintf (fp, "P6\n%u %u\n255\n", width / 2, height / 2);
        fwrite (cuda_out_buffer, 1, width * height * 3 / 2 / 2, fp);
        fclose (fp);
    }
    */

#ifdef opencv
    //cv::cuda::GpuMat d_mat(height / 2, width / 2, CV_8UC3, cuda_out_buffer);
    cv::cuda::GpuMat d_16mat(height, width, CV_16UC3, cuda_out_buffer);
    cv::cuda::GpuMat d_mat;
    d_16mat.convertTo(d_mat, CV_8UC3, 1/64.0);

#ifdef CALIB
    cv::Mat gray;
    if(count % 10 == 0) {
            //cv::Mat mat(height  / 2, width / 2, CV_8UC3, cuda_out_buffer);
            cv::Mat mat(height, width, CV_16UC3, cuda_out_buffer);
	    cv::cvtColor(mat, gray, cv::COLOR_BGRA2GRAY);

	    bool found = false;
	    found = cv::findChessboardCorners(mat, board_size, corners,
					      cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FILTER_QUADS);
	    if (found)
	    {
	      cv::cornerSubPix(gray, corners, cv::Size(5, 5), cv::Size(-1, -1),
			   cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, 30, 0.1));
	      cv::drawChessboardCorners(gray, board_size, corners, found);

	      std::vector< cv::Point3f > obj;
	      for (int i = 0; i < board_height; i++)
		for (int j = 0; j < board_width; j++)
		  obj.push_back(cv::Point3f((float)j * square_size, (float)i * square_size, 0));

	      std::cout << count << ". Found corners!" << std::endl;
	      image_points.push_back(corners);
	      object_points.push_back(obj);
	    }
	    else std::cout << count << ". no corners found" << std::endl;
	    cv::imshow("chessboard", gray);
    }
    if(count == 1000) {
	  std::cout << "Starting Calibration" << std::endl;
	  cv::Mat K;
	  cv::Mat D;
	  std::vector< cv::Mat > rvecs, tvecs;
	  int flag = 0;
	  flag |= cv::CALIB_FIX_K4;
	  flag |= cv::CALIB_FIX_K5;
	  cv::calibrateCamera(object_points, image_points, gray.size(), K, D, rvecs, tvecs, flag);
	  std::cout << "K" << K << std::endl;
          std::cout << "D" << D << std::endl;
  	  std::cout << "board_width" << board_width << std::endl;
      	  std::cout << "board_height" << board_height << std::endl;
  	  std::cout << "square_size" << square_size << std::endl;
	  std::cout << "Done Calibration" << std::endl << std::endl;
    }
#endif

#ifdef APRILTAGS
    uint32_t num_detections;
    //input_image.dev_ptr = (uchar3*)cuda_out_buffer;
    input_image.dev_ptr = (uchar3*) d_mat.data;
    //input_image.pitch = d_mat.step;
    input_image.pitch = width*3;
    std::cout << input_image.pitch << std::endl;
    cudaStreamAttachMemAsync(main_stream, input_image.dev_ptr, 0, cudaMemAttachGlobal);
    const int error = cuAprilTagsDetect(
      april_tags_handle, &input_image, tags.data(),
      &num_detections, max_tags, main_stream);
    cudaStreamAttachMemAsync(main_stream, input_image.dev_ptr, 0, cudaMemAttachHost);
    cudaStreamSynchronize(main_stream);

    if(error != 0) {
	    std::cout << "april tag detect error" << std::endl;
    }
    
    if (num_detections > 0) {
    	std::cout << "frame " << count << " found tag";
    }
    for (uint32_t i = 0; i < num_detections; i++) {
       const cuAprilTagsID_t & detection = tags[i];

       std::cout << " " <<  detection.id << ":" << detection.translation[0];
       std::cout << "," << detection.translation[1];
       std::cout << "," << detection.translation[2] << " ";

       const Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::ColMajor>>
            orientation(detection.orientation);
       const Eigen::Quaternion<float> q(orientation);
       std::cout << "quaternion: " << q.w() << " " << q.x() << " " << q.y() << " " << q.z() << std::endl;
       const Eigen::AngleAxis<float> axis(q);
       std::cout << "axis: " << axis.axis() << std::endl;

    }
    if (num_detections > 0)
    	std::cout << std::endl;

#endif

    if(count % 10 == 0) {
        std::string str = "fps: " + std::to_string(fps);
	if(true) {
		cv::Mat mat(d_mat);
		//cv::Mat mat(height / 2, width / 2, CV_8UC3, cuda_out_buffer);
		cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR);
#ifdef APRILTAGS
		for (uint32_t i = 0; i < num_detections; i++) {
			const cuAprilTagsID_t & detection = tags[i];

			cv::Point tag_points[4];
		       tag_points[0] = cv::Point( detection.corners[0].x, detection.corners[0].y);
		       tag_points[1] = cv::Point( detection.corners[1].x, detection.corners[1].y);
		       tag_points[2] = cv::Point( detection.corners[2].x, detection.corners[2].y);
		       tag_points[3] = cv::Point( detection.corners[3].x, detection.corners[3].y);
		       cv::line(mat, tag_points[0], tag_points[1], cv::Scalar(255,0,0), 2);
		       cv::line(mat, tag_points[1], tag_points[2], cv::Scalar(255,0,0), 2);
		       cv::line(mat, tag_points[2], tag_points[3], cv::Scalar(255,0,0), 2);
		       cv::line(mat, tag_points[3], tag_points[0], cv::Scalar(255,0,0), 2);
		}
#endif
		cv::putText(mat, str, cv::Point(50,50),cv::FONT_HERSHEY_DUPLEX,1,cv::Scalar(0,255,0),2,false);
		cv::imshow(dev_name, mat);
	}
        cv::pollKey();
	std::cout << str << std::endl;
    }
#endif
}

static int
read_frame                      (double fps)
{
    struct v4l2_buffer buf;
    unsigned int i;

    CLEAR (buf);

    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_USERPTR;

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

    process_image ((void *) buf.m.userptr, fps);

    if (-1 == xioctl (fd, VIDIOC_QBUF, &buf))
	errno_exit ("VIDIOC_QBUF");

    return 1;
}

static void
mainloop                        (void)
{
    time_t start, end;
    double seconds = 0;
    double fps  = 0;

    while(true) {
	    count = 0;
	    time(&start);
	    while (count < 60) {
		for (;;) {
		    fd_set fds;
		    struct timeval tv;
		    int r;

		    FD_ZERO (&fds);
		    FD_SET (fd, &fds);

		    /* Timeout. */
		    tv.tv_sec = 10;
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

		    count++;
		    if (read_frame (fps))
			break;

		    /* EAGAIN - continue select loop. */
		}
	    }

	    time(&end);
	    seconds = difftime (end, start);
	    fps  = (double) count / seconds;
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
    control.value = 1;
    if (-1 == xioctl (fd, VIDIOC_S_CTRL, &control)) {
        errno_exit ("VIDIOC_S_CTRL low latency mode");
    }

    CLEAR(ext_controls);
    CLEAR(ext_control);
    ext_controls.ctrl_class = V4L2_CTRL_ID2CLASS(0x009a2000);
    ext_controls.count = 1;
    ext_controls.controls = &ext_control;

    ext_control.id = 0x009a2008; // sensor_mode
//    ext_control.value64 = 0;
//    ext_control.value64 = 1;
//    ext_control.value64 = 2;
//    ext_control.value64 = 3;
    ext_control.value64 = 4;
    if (-1 == xioctl (fd, VIDIOC_S_EXT_CTRLS, &ext_controls)) {
        errno_exit ("VIDIOC_S_CTRL sensor mode");
    }

    ext_control.id = 0x009a200b; // FRAME_RATE
//    ext_control.value64 = 120000000; 
    ext_control.value64 = 60000000; 
//    ext_control.value64 = 30000000; 
    if (-1 == xioctl (fd, VIDIOC_S_EXT_CTRLS, &ext_controls)) {
        errno_exit ("VIDIOC_S_CTRL frame rate");
    }

    ext_control.id = 0x009a2009; // gain
    ext_control.value64 = 16; 
   // ext_control.value64 = 1000; 
    if (-1 == xioctl (fd, VIDIOC_S_EXT_CTRLS, &ext_controls)) {
        errno_exit ("VIDIOC_S_CTRL gain");
    }

    ext_control.id = 0x009a200a; // exposure
//    ext_control.value64 = 100; 
    ext_control.value64 = 200000; 
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

#ifdef APRILTAGS
    const int error = nvCreateAprilTagsDetector(
//      &april_tags_handle, width/2, height/2, tile_size, cuAprilTagsFamily::NVAT_TAG16H5,
      &april_tags_handle, width, height, tile_size, cuAprilTagsFamily::NVAT_TAG16H5,
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
    //input_image_buffer_size = width * height * 3 / 2 / 2 * sizeof(char);
    //input_image.width = width/2;
    //input_image.height = height/2;
    input_image_buffer_size = width * height * 3 * sizeof(char);
    input_image.width = width;
    input_image.height = height;
    //input_image.pitch = 16; // not sure
    // set input_image.dev_ptr to buffer before detecting
#endif
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
//    size_t size = width * height * 3 / 2 / 2 * sizeof(char);
//    size_t size = width * height * 3 * sizeof(char);
//    size_t size = width * height * 3 / 2 / 2 * sizeof(short);
    size_t size = width * height * 3 * sizeof(short);
    cudaMallocManaged (&cuda_out_buffer, size, cudaMemAttachGlobal);

    cudaDeviceSynchronize ();
}

static void
usage                           (FILE *                 fp,
                                 int                    argc,
                                 char **                argv)
{
    fprintf (fp,
            "Usage: %s [options]\n\n"
            "Options:\n"
            "-c | --count N       Frame count (default: %u)\n"
            "-d | --device name   Video device name (default: %s)\n"
            "-f | --format        Capture input pixel format (default: RG10)\n"
            "-h | --help          Print this message\n"
            "-o | --output        Output file name (default: %s)\n"
            "-s | --size WxH      Frame size (default: %ux%u)\n"
            "Experimental options:\n"
            "-F | --field         Capture field (default: none)\n"
            "",
            argv[0], count, dev_name, file_name, width, height);
}

static const char short_options [] = "c:d:f:F:hmo:rs:uz";

static const struct option
long_options [] = {
    { "count",      required_argument,      NULL,           'c' },
    { "device",     required_argument,      NULL,           'd' },
    { "format",     required_argument,      NULL,           'f' },
    { "field",      required_argument,      NULL,           'F' },
    { "help",       no_argument,            NULL,           'h' },
    { "output",     required_argument,      NULL,           'o' },
    { "size",       required_argument,      NULL,           's' },
    { 0, 0, 0, 0 }
};

static struct {
    const char *name;
    unsigned int fourcc;
} pixel_formats[] = {
    { "RGB332", V4L2_PIX_FMT_RGB332 },
    { "RGB555", V4L2_PIX_FMT_RGB555 },
    { "RGB565", V4L2_PIX_FMT_RGB565 },
    { "RGB555X", V4L2_PIX_FMT_RGB555X },
    { "RGB565X", V4L2_PIX_FMT_RGB565X },
    { "BGR24", V4L2_PIX_FMT_BGR24 },
    { "RGB24", V4L2_PIX_FMT_RGB24 },
    { "BGR32", V4L2_PIX_FMT_BGR32 },
    { "RGB32", V4L2_PIX_FMT_RGB32 },
    { "Y8", V4L2_PIX_FMT_GREY },
    { "Y10", V4L2_PIX_FMT_Y10 },
    { "Y12", V4L2_PIX_FMT_Y12 },
    { "Y16", V4L2_PIX_FMT_Y16 },
    { "UYVY", V4L2_PIX_FMT_UYVY },
    { "VYUY", V4L2_PIX_FMT_VYUY },
    { "YUYV", V4L2_PIX_FMT_YUYV },
    { "YVYU", V4L2_PIX_FMT_YVYU },
    { "NV12", V4L2_PIX_FMT_NV12 },
    { "NV21", V4L2_PIX_FMT_NV21 },
    { "NV16", V4L2_PIX_FMT_NV16 },
    { "NV61", V4L2_PIX_FMT_NV61 },
    { "NV24", V4L2_PIX_FMT_NV24 },
    { "NV42", V4L2_PIX_FMT_NV42 },
    { "SBGGR8", V4L2_PIX_FMT_SBGGR8 },
    { "SGBRG8", V4L2_PIX_FMT_SGBRG8 },
    { "SGRBG8", V4L2_PIX_FMT_SGRBG8 },
    { "SRGGB8", V4L2_PIX_FMT_SRGGB8 },
    { "SBGGR10_DPCM8", V4L2_PIX_FMT_SBGGR10DPCM8 },
    { "SGBRG10_DPCM8", V4L2_PIX_FMT_SGBRG10DPCM8 },
    { "SGRBG10_DPCM8", V4L2_PIX_FMT_SGRBG10DPCM8 },
    { "SRGGB10_DPCM8", V4L2_PIX_FMT_SRGGB10DPCM8 },
    { "SBGGR10", V4L2_PIX_FMT_SBGGR10 },
    { "SGBRG10", V4L2_PIX_FMT_SGBRG10 },
    { "SGRBG10", V4L2_PIX_FMT_SGRBG10 },
    { "SRGGB10", V4L2_PIX_FMT_SRGGB10 },
    { "SBGGR12", V4L2_PIX_FMT_SBGGR12 },
    { "SGBRG12", V4L2_PIX_FMT_SGBRG12 },
    { "SGRBG12", V4L2_PIX_FMT_SGRBG12 },
    { "SRGGB12", V4L2_PIX_FMT_SRGGB12 },
    { "DV", V4L2_PIX_FMT_DV },
    { "MJPEG", V4L2_PIX_FMT_MJPEG },
    { "MPEG", V4L2_PIX_FMT_MPEG },
    { "RG10", V4L2_PIX_FMT_SRGGB10 },
};

static unsigned int v4l2_format_code(const char *name)
{
    unsigned int i;

    for (i = 0; i < ARRAY_SIZE(pixel_formats); ++i) {
        if (strcasecmp(pixel_formats[i].name, name) == 0)
            return pixel_formats[i].fourcc;
    }

    return 0;
}

static struct {
    const char *name;
    unsigned int field;
} fields[] = {
    { "ANY", V4L2_FIELD_ANY },
    { "NONE", V4L2_FIELD_NONE },
    { "TOP", V4L2_FIELD_TOP },
    { "BOTTOM", V4L2_FIELD_BOTTOM },
    { "INTERLACED", V4L2_FIELD_INTERLACED },
    { "SEQ_TB", V4L2_FIELD_SEQ_TB },
    { "SEQ_BT", V4L2_FIELD_SEQ_BT },
    { "ALTERNATE", V4L2_FIELD_ALTERNATE },
    { "INTERLACED_TB", V4L2_FIELD_INTERLACED_TB },
    { "INTERLACED_BT", V4L2_FIELD_INTERLACED_BT },
};

static unsigned int v4l2_field_code(const char *name)
{
    unsigned int i;

    for (i = 0; i < ARRAY_SIZE(fields); ++i) {
        if (strcasecmp(fields[i].name, name) == 0)
            return fields[i].field;
    }

    return -1;
}

int
main                            (int                    argc,
                                 char **                argv)
{
    std::cout << cv::getBuildInformation();

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

            case 'c':
                count = atoi (optarg);
                break;

            case 'd':
                dev_name = optarg;
                break;

            case 'f':
                pixel_format = v4l2_format_code(optarg);
                if (pixel_format == 0) {
                    printf("Unsupported video format '%s'\n", optarg);
                    pixel_format = V4L2_PIX_FMT_UYVY;
                }
                break;

            case 'F':
                field = v4l2_field_code(optarg);
                if ((int)field < 0) {
                    printf("Unsupported field '%s'\n", optarg);
                    field = V4L2_FIELD_INTERLACED;
                }
                break;

            case 'h':
                usage (stdout, argc, argv);
                exit (EXIT_SUCCESS);

            case 'o':
                file_name = optarg;
                break;

            case 's':
                width = atoi (strtok (optarg, "x"));
                height = atoi (strtok (NULL, "x"));
                break;

            default:
                usage (stderr, argc, argv);
                exit (EXIT_FAILURE);
        }
    }

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
