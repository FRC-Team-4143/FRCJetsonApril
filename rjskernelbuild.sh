#get kernel sources from https://developer.nvidia.com/downloads/embedded/l4t/r35_release_v3.1/sources/public_sources.tbz2/


export LOCALVERSION=-test
export TEGRA_KERNEL_OUT=/media/ubuntu/673fe9d8-6f00-4487-b046-8ccee8268a74/tx2nx/build
export TEGRA_MODULES_OUT=/media/ubuntu/673fe9d8-6f00-4487-b046-8ccee8268a74/tx2nx/modsout/
#make ARCH=arm64 O=$TEGRA_KERNEL_OUT tegra_defconfig
make ARCH=arm64 O=$TEGRA_KERNEL_OUT -j4
make ARCH=arm64 O=$TEGRA_KERNEL_OUT modules_install INSTALL_MOD_PATH=${TEGRA_MODULES_OUT}/
sudo cp -R ${TEGRA_MODULES_OUT}/lib/modules/4.9.253-test /lib/modules
sudo cp /media/ubuntu/673fe9d8-6f00-4487-b046-8ccee8268a74/tx2nx/build/arch/arm64/boot/dts/tegra186-p3636-0001-p3509-0000-a01.dtb /boot/test.dtb
sudo cp $TEGRA_KERNEL_OUT/arch/arm64/boot/Image /boot/Image-4.9.253-test

