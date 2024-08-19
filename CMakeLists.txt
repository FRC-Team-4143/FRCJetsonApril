cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
#  set(CMAKE_CUDA_ARCHITECTURES 87)
#endif()

#add_subdirectory( frc971/orin/abseil-cpp )

project(capture LANGUAGES CXX)
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
#find_package(wpilib REQUIRED)   # this is broken RJS

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip
)
FetchContent_MakeAvailable(googletest)

# https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip



add_compile_options(-Wno-psabi)

include_directories( /usr/local/include/cameraserver /usr/local/include/cscore /usr/local/include/wpiutil
	/usr/local/include/wpimath /usr/local/include/ntcore /usr/local/include/apriltag )
#frc971/orin/cub 

cuda_add_executable( captureusb captureusb.cpp yuv2rgb.cu )
target_link_libraries( captureusb ${OpenCV_LIBS} ${CMAKE_SOURCE_DIR}/libcuapriltags.a )

cuda_add_executable( capturestockrpiv2 capturestockrpiv2.cpp raw2rgb.cu )
target_link_libraries( capturestockrpiv2 nppicc ${OpenCV_LIBS} ${CMAKE_SOURCE_DIR}/libcuapriltags.a )

cuda_add_executable( calibrate calibrate.cpp raw2rgb.cu )
target_link_libraries( calibrate ${OpenCV_LIBS} )

cuda_add_executable( capture-cuda capture.cpp raw2rgb.cu )
target_link_libraries( capture-cuda ${OpenCV_LIBS} ${CMAKE_SOURCE_DIR}/libcuapriltags.a 
	cscore ntcore apriltag wpimath wpiutil )

cuda_add_executable( capture-cuda1 capture.cpp raw2rgb.cu )
target_compile_definitions( capture-cuda1 PUBLIC DEVICE1 )
target_link_libraries( capture-cuda1 ${OpenCV_LIBS} ${CMAKE_SOURCE_DIR}/libcuapriltags.a 
	cscore ntcore apriltag wpimath wpiutil )


add_subdirectory(abseil-cpp)

cuda_add_executable( frc971orin 
	frc971/orin/apriltag.cu 
	frc971/orin/apriltag_detect.cu 
	frc971/orin/gpu_apriltag.cu
	frc971/orin/labeling_allegretti_2019_BKE.cu
	frc971/orin/line_fit_filter.cu
	frc971/orin/main.cu
	frc971/orin/points.cu
	frc971/orin/threshold.cu
	frc971/orin/cuda.cc
	)
target_include_directories(frc971orin PUBLIC . )
target_link_libraries(frc971orin GTest::gtest ${OpenCV_LIBS} absl::flags absl::log )
