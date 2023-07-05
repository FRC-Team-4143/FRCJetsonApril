Install on SSD with instructions here:
https://jetsonhacks.com/2023/05/30/jetson-orin-nano-tutorial-ssd-install-boot-and-jetpack-setup/

Compile OpenCV with CUDA support with instructions here:
https://github.com/mdegans/nano_build_opencv
./build_opencv.sh 4.8.0

jtop should show something like this on INFO tab:
OpenCV: 4.8.0 with CUDA: YES


Get NVIDIA apriltag library from:
https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_nitros/blob/main/isaac_ros_nitros/lib/cuapriltags/lib_aarch64_jetpack51/libcuapriltags.a

bspatch libcuapriltags.a libcuapriltags16h5.a libcuapriltags16h5.a.bspatch


see orinnanooverlaycmd.sh to set kernel overlay.


capturestockrpiv2 should read 16H5 tags. tested with both RPIv2 camera and an Arducam IMX219 on a Jetson Orin Nano devkit with stock kernel on JetPack 5.1.1.







#THE FOLLOWING LIBRARIES DO NOT SUPPORT ORIN

https://github.com/NVIDIA-AI-IOT/isaac_ros_apriltag/blob/main/isaac_ros_apriltag/nvapriltags/lib_aarch64_jetpack44/libapril_tagging.a

--- and ---

https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_apriltag/blob/release-ea3/isaac_ros_apriltag/nvapriltags/lib_aarch64_jetpack44/libapril_tagging.a

