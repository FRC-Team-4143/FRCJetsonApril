# tar xf public_sources.tbz2 
# cd Linux_for_Tegra/source
# tar xf kernel_src.tbz2
# tar xf kernel_oot_modules_src.tbz2
# tar xf nvidia_kernel_display_driver_source.tbz2

# apply patch


cd ~/Linux_for_Tegra/source/kernel
make -j
sudo make install

cd ~/Linux_for_Tegra/source
make -j modules
make dtbs
sudo make modules_install
sudo cp nvidia-oot/device-tree/platform/generic-dts/dtbs/* /boot
sudo nv-update-initrd

