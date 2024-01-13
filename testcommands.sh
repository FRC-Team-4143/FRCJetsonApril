# doesn't work
gst-launch-1.0 -v v4l2src device=/dev/video0 ! video/x-raw,framerate=60/1,width=1280,height=720 ! nvvidconv ! 'video/x-raw(memory:NVMM), format=NV12' ! nv3dsink


#works
gst-launch-1.0 nvarguscamerasrc ! 'video/x-raw(memory:NVMM),width=1280,height=720,framerate=60/1,format=NV12' ! nv3dsink

#works
gst-launch-1.0 nvarguscamerasrc ! 'video/x-raw(memory:NVMM),width=1920,height=1080,framerate=30/1,format=NV12' ! nv3dsink

#doesn't work
v4l2-ctl -d /dev/video0 --set-fmt-video=width=1920,height=1080,pixelformat=RG10 --stream-count=1 --stream-mmap --stream-to=dev0 --verbose

gst-launch-1.0 -v v4l2src device="/dev/video0" ! video/x-raw,framerate=30/1,width=1920,height=1080,format=UYVY ! xvimagesink &

gst-launch-1.0 nvarguscamerasrc sensor-id=0 ! 'video/x-raw(memory:NVMM),width=1280, height=720, framerate=30/1, format=NV12' ! nvvidconv flip-method=0 ! 'video/x-raw,width=1280, height=720' ! nvvidconv ! nvegltransform ! nveglglessink -e

sudo /opt/nvidia/jetson-io/jetson-io.py

v4l2-ctl -d /dev/video0 --set-fmt-video=width=1280,height=720,pixelformat=RG10 --set-ctrl bypass_mode=0 --stream-mmap --stream-count=300



v4l2-ctl --set-fmt-video=width=1280,height=800,pixelformat=RG10 --set-ctrl bypass_mode=0 --stream-mmap --stream-count=1 -d /dev/video1 --stream-to=capture.raw
