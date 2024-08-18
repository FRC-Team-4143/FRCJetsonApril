Currently setup for 36H11 tags and Innovision ov9281 camera

Optional install on SSD with instructions here:
https://jetsonhacks.com/2023/05/30/jetson-orin-nano-tutorial-ssd-install-boot-and-jetpack-setup/
Or use SD card image

Compile OpenCV with CUDA support with instructions here:
https://github.com/mdegans/nano_build_opencv
./build_opencv.sh 4.8.0

jtop should show something like this on INFO tab:
OpenCV: 4.8.0 with CUDA: YES

Get NVIDIA apriltag library from:
https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_nitros/blob/main/isaac_ros_nitros/lib/cuapriltags/lib_aarch64_jetpack51/libcuapriltags.a   committed on Aug 1,2023 commit dc959fb

download kernel sources and apply patch for Innovision ov9281 camera.
kernel sources are here https://developer.nvidia.com/downloads/embedded/l4t/r35_release_v3.1/sources/public_sources.tbz2/
apply orinov9281working8bitkernel35.3.1.patch to nvidia kernel sources
run rjsorinkernelbuild.sh to build and install -tegra4143
modify /boot/extlinux/extlinux.conf to allow custom kernel booting

build and install https://github.com/wpilibsuite/allwpilib from source
cmake .. -DWITH_JAVA=OFF -DWITH_GUI=OFF -DWITH_EXAMPLES=OFF -DWITH_TESTS=OFF

use cmake to build capture-cuda

must run sudo ./debugcamera.sh to increase clock speeds on every boot see rc.local

capture-cuda is setup for 36H11 tags and Innovision camera ov9281 with custom kernel





#THE FOLLOWING LIBRARIES DO NOT SUPPORT ORIN

https://github.com/NVIDIA-AI-IOT/isaac_ros_apriltag/blob/main/isaac_ros_apriltag/nvapriltags/lib_aarch64_jetpack44/libapril_tagging.a

--- and ---

https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_apriltag/blob/release-ea3/isaac_ros_apriltag/nvapriltags/lib_aarch64_jetpack44/libapril_tagging.a

