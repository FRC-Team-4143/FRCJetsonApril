cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 87)
endif()

if(NOT DEFINED CMAKE_CUDA20_STANDARD_COMPILE_OPTION)
  set(CMAKE_CUDA20_STANDARD_COMPILE_OPTION "-std=c++20")
  set(CMAKE_CUDA20_EXTENSION_COMPILE_OPTION "-std=c++20")
endif()

set(CMAKE_CUDA_FLAGS "--expt-relaxed-constexpr")

project(capture LANGUAGES CXX CUDA)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
#find_package(wpilib REQUIRED)   # this is broken RJS

add_compile_options(-Wno-psabi)

include_directories( . /usr/local/cuda/include /usr/local/include/cameraserver 
	/usr/local/include/cscore /usr/local/include/wpiutil
	/usr/local/include/wpimath /usr/local/include/ntcore 
	/usr/local/include/apriltag )

add_executable( captureusb captureusb.cu yuv2rgb.cu 
	frc971/orin/971apriltag.cu 
	frc971/orin/apriltag_detect.cu 
	frc971/orin/labeling_allegretti_2019_BKE.cu
	frc971/orin/line_fit_filter.cu
	frc971/orin/points.cu
	frc971/orin/threshold.cu
	frc971/orin/cuda.cc
	)
target_link_libraries( captureusb ${OpenCV_LIBS} ${CMAKE_SOURCE_DIR}/libcuapriltags.a ${CMAKE_SOURCE_DIR}/third_party/apriltag/build/libapriltag.a ) 

add_executable( capturestockrpiv2 capturestockrpiv2.cpp raw2rgb.cu )
target_link_libraries( capturestockrpiv2 nppicc ${OpenCV_LIBS} ${CMAKE_SOURCE_DIR}/libcuapriltags.a )

add_executable( calibrate calibrate.cpp raw2rgb.cu )
target_link_libraries( calibrate ${OpenCV_LIBS} )

add_executable( capture-cuda capture.cpp raw2rgb.cu )
target_link_libraries( capture-cuda ${OpenCV_LIBS} ${CMAKE_SOURCE_DIR}/libcuapriltags.a 
	cscore ntcore apriltag wpimath wpiutil )

add_executable( capture-cuda1 capture.cpp raw2rgb.cu )
target_compile_definitions( capture-cuda1 PUBLIC DEVICE1 )
target_link_libraries( capture-cuda1 ${OpenCV_LIBS} ${CMAKE_SOURCE_DIR}/libcuapriltags.a 
	cscore ntcore apriltag wpimath wpiutil )

#add_subdirectory(third_party/apriltag)
