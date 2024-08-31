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

#include <asm/types.h>          /* for videodev2.h */

#include <iostream>
#include <time.h>
#include <linux/videodev2.h>

#include <cuda_runtime.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <opencv2/calib3d.hpp>
std::vector< std::vector< cv::Point3f > > object_points;
std::vector< std::vector< cv::Point2f > > image_points;
std::vector< cv::Point2f > corners;

int board_width = 7;
int board_height = 5;

cv::Size board_size = cv::Size(board_width, board_height);
int board_n = board_width * board_height;
float square_size = 0.028;  // in meters?

#include "raw2rgb.cuh"

#define CLEAR(x) memset (&(x), 0, sizeof (x))
#define ARRAY_SIZE(a)   (sizeof(a)/sizeof((a)[0]))

struct buffer {
    void *                  start;
    size_t                  length;
};

static const char *     dev_name        = "/dev/video0";
static int              fd              = -1;
struct buffer *         buffers         = NULL;
static unsigned int     n_buffers       = 0;
static unsigned int     width           = 1280; // mode 4 or 5
static unsigned int     height          = 720;
static unsigned int     count           = 0;
static unsigned char *  cuda_out_buffer = NULL;
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
    //gpuConvertrawtoRGB ((unsigned short *) p, cuda_out_buffer, width, height);
    //gpuConvertrawtoRGB ((unsigned char *) p, cuda_out_buffer, width, height);
    //gpuConvertrawtoRGBA ((unsigned char *) p, cuda_out_buffer, width, height);
    //gpuConvertgraytoRGB ((unsigned char *) p, cuda_out_buffer, width, height);
    gpuConvertgraytoRGB ((unsigned short *) p, cuda_out_buffer, width, height, 0);
    cudaStreamSynchronize(0);

    cv::Mat gray;
    cv::Mat mat(height, width, CV_8UC3, cuda_out_buffer);
    cv::cvtColor(mat, gray, cv::COLOR_BGRA2GRAY);

    if(count % 10 == 0) {
	    bool found = false;
	    found = cv::findChessboardCorners(gray, board_size, corners,
					      cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FILTER_QUADS);
	    if (found)
	    {
	      cv::cornerSubPix(gray , corners, cv::Size(5, 5), cv::Size(-1, -1),
			   cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, 30, 0.1));
	      cv::drawChessboardCorners(gray , board_size, corners, found);

	      std::vector< cv::Point3f > obj;
	      for (int i = 0; i < board_height; i++)
		for (int j = 0; j < board_width; j++)
		  obj.push_back(cv::Point3f((float)j * square_size, (float)i * square_size, 0));

	      std::cout << count << ". Found corners!" << std::endl;
	      image_points.push_back(corners);
	      object_points.push_back(obj);
	    }
	    else std::cout << count << ". no corners found" << std::endl;
	    cv::imshow("chessboard", gray );
	    cv::pollKey();
    }
    if(count == 1000) {
	  std::cout << "Starting Calibration" << std::endl;
	  cv::Mat K;
	  cv::Mat D;
	  std::vector< cv::Mat > rvecs, tvecs;
	  int flag = 0;
	  flag |= cv::CALIB_FIX_K4;
	  flag |= cv::CALIB_FIX_K5;
	  cv::calibrateCamera(object_points, image_points, mat.size(), K, D, rvecs, tvecs, flag);
	  std::cout << "K" << K << std::endl;
          std::cout << "D" << D << std::endl;
  	  std::cout << "board_width" << board_width << std::endl;
      	  std::cout << "board_height" << board_height << std::endl;
  	  std::cout << "square_size" << square_size << std::endl;
	  std::cout << "Done Calibration" << std::endl << std::endl;
	  exit(0);
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
	    while (count < 1200) {
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
    ext_control.value64 = 0;
//    ext_control.value64 = 1;
//    ext_control.value64 = 4;
//    ext_control.value64 = 3;
//    if (-1 == xioctl (fd, VIDIOC_S_EXT_CTRLS, &ext_controls)) {
//        errno_exit ("VIDIOC_S_CTRL sensor mode");
//    }

    ext_control.id = 0x009a200b; // FRAME_RATE
    ext_control.value64 = 120000000; 
//    if (-1 == xioctl (fd, VIDIOC_S_EXT_CTRLS, &ext_controls)) {
//        errno_exit ("VIDIOC_S_CTRL frame rate");
//    }

    ext_control.id = 0x009a2009; // gain
    ext_control.value64 = 10000; 
    //ext_control.value64 = 0; 
    //if (-1 == xioctl (fd, VIDIOC_S_EXT_CTRLS, &ext_controls)) {
    //   errno_exit ("VIDIOC_S_CTRL gain");
    //}

    ext_control.id = 0x009a200a; // exposure
    ext_control.value64 = 100; 
    //ext_control.value64 = 0; 
//    if (-1 == xioctl (fd, VIDIOC_S_EXT_CTRLS, &ext_controls)) {
//        errno_exit ("VIDIOC_S_CTRL exposure");
//    }

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
//      &april_tags_handle, width/2, height/2, cuAprilTagsFamily::NVAT_TAG36H11,
      &april_tags_handle, width, height, tile_size, cuAprilTagsFamily::NVAT_TAG36H11,
//      &april_tags_handle, width, height, cuAprilTagsFamily::NVAT_TAG16H5,
      &cam_intrinsics, tag_size);
    if (error != 0) {
      throw std::runtime_error(
              "Failed to create NV April Tags detector (error code " +
              std::to_string(error) + ")");
    }

    // Allocate the output vector to contain detected AprilTags.
    tags.resize(max_tags);
//    input_image.width = width/2;
//    input_image.height = height/2;
    input_image.width = width;
    input_image.height = height;
    input_image.pitch = 3; // not sure
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
    size_t size = width * height * 3 * sizeof(char);
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
