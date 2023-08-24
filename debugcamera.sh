#!/bin/sh

nvpmodel -m 0
jetson_clocks
jetson_clocks --show

echo 1 > /sys/kernel/debug/bpmp/debug/clk/vi/mrq_rate_locked
echo 1 > /sys/kernel/debug/bpmp/debug/clk/isp/mrq_rate_locked
echo 1 > /sys/kernel/debug/bpmp/debug/clk/nvcsi/mrq_rate_locked
cat /sys/kernel/debug/bpmp/debug/clk/vi/max_rate |tee /sys/kernel/debug/bpmp/debug/clk/vi/rate
cat /sys/kernel/debug/bpmp/debug/clk/isp/max_rate | tee  /sys/kernel/debug/bpmp/debug/clk/isp/rate
cat /sys/kernel/debug/bpmp/debug/clk/nvcsi/max_rate | tee /sys/kernel/debug/bpmp/debug/clk/nvcsi/rate
#cd /sys/kernel/debug/dynamic_debug/
#echo file csi2_fops.c +p > control
#echo 1 > /sys/kernel/debug/tracing/tracing_on
#echo 30720 > /sys/kernel/debug/tracing/buffer_size_kb
#echo 1 > /sys/kernel/debug/tracing/events/tegra_rtcpu/enable
#echo 1 > /sys/kernel/debug/tracing/events/freertos/enable
#echo 2 > /sys/kernel/debug/camrtc/log-level
#echo 1 > /sys/kernel/debug/tracing/events/camera_common/enable
#echo > /sys/kernel/debug/tracing/trace
#cat /sys/kernel/debug/tracing/trace
