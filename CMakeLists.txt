cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
#  set(CMAKE_CUDA_ARCHITECTURES 87)
#endif()

project(capture LANGUAGES CXX CUDA )
find_package(OpenCV REQUIRED)
#find_package(wpilib REQUIRED)   # this is broken RJS

add_compile_options(-Wno-psabi)

include_directories( /usr/local/include/cameraserver /usr/local/include/cscore /usr/local/include/wpiutil
	/usr/local/include/wpimath /usr/local/include/ntcore /usr/local/include/apriltag )

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
