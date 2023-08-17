export LOCALVERSION=-tegra4143
export TEGRA_KERNEL_OUT=/home/ubuntu/kernel35.3.1/build
export TEGRA_MODULES_OUT=/home/ubuntu/kernel35.3.1/modsout
cd /home/ubuntu/kernel35.3.1/orinmod/kernel/kernel-5.10
#make ARCH=arm64 O=$TEGRA_KERNEL_OUT tegra_defconfig
make ARCH=arm64 O=$TEGRA_KERNEL_OUT -j5 --output-sync=target Image
make ARCH=arm64 O=$TEGRA_KERNEL_OUT -j5 --output-sync=target dtbs
make ARCH=arm64 O=$TEGRA_KERNEL_OUT -j5 --output-sync=target modules
make ARCH=arm64 O=$TEGRA_KERNEL_OUT modules_install INSTALL_MOD_PATH=${TEGRA_MODULES_OUT}/

cd /home/ubuntu/kernel35.3.1/NVIDIA-kernel-module-source-TempVersion
make ARCH=arm64 O=$TEGRA_KERNEL_OUT -j5 modules
make ARCH=arm64 O=$TEGRA_KERNEL_OUT modules_install INSTALL_MOD_PATH=${TEGRA_MODULES_OUT}/

sudo cp -R ${TEGRA_MODULES_OUT}/lib/modules/5.10.104-tegra4143 /lib/modules
sudo cp $TEGRA_KERNEL_OUT/arch/arm64/boot/dts/nvidia/tegra234-p3767-camera-ov9281-dual.dtbo /boot
sudo cp $TEGRA_KERNEL_OUT/arch/arm64/boot/dts/nvidia/tegra234-p3767-camera-p3768-ov9281-dual.dtbo /boot
sudo cp $TEGRA_KERNEL_OUT/arch/arm64/boot/dts/nvidia/tegra234-p3767-0003-p3768-0000-a0.dtb /boot/dtb
sudo cp $TEGRA_KERNEL_OUT/arch/arm64/boot/Image /boot/Image-5.10.104-tegra4143
