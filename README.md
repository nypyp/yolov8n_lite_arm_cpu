# README

## yolov8 fall detect

系统版本：Firfly固件AIO-3399J_Ubuntu18.04-Minimal-r240_v2.5.1d_230330

设备：Firfly RK3390J

‍

## Installation

### 安装PaddleLite库文件

[PaddleLite 预编译库v2.13-rc](https://github.com/PaddlePaddle/Paddle-Lite/releases/tag/v2.13-rc)  
下载预编译库，选择armlinux.armv8.gcc.with_extra.with_cv版本

```bash
wget https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.13-rc/inference_lite_lib.armlinux.armv8.gcc.with_extra.with_cv.tar.gz
```

在demo/cxx路径下放置项目：

```bash
tar -xzjv inference_lite_lib.armlinux.armv8.gcc.with_extra.with_cv.tar.gz
cd inference_lite_lib.armlinux.armv8.gcc.with_extra.with_cv/demo/cxx/
git clone https://github.com/nypyp/yolov8n_lite_arm_cpu.git
cd yolov8n_lite_arm_cpu
git submodule update --init --recursive #递归初始化子仓库
```

### 安装编译工具

```bash
sudo apt-get update
sudo apt-get install build-essential gcc make cmake git libopencv-dev 
```

### 安装Paho Mqtt C++库

```bash
#递归克隆主仓库以及子仓库，即Paho mqtt c库，编译需要
$ cd yolov8n_lite_arm_cpu/external/paho.mqtt.cpp
$ sudo apt-get insatll libssl-dev

$ cmake -Bbuild -H. \
-DPAHO_WITH_MQTT_C=ON \
-DPAHO_BUILD_STATIC=ON \
-DPAHO_BUILD_SAMPLES=OFF
$ sudo cmake --build build/ --target install  
$ sudo ldconfig
```

### 安装Jsoncpp库

```bash
$ cd yolov8n_lite_arm_cpu/external/jsoncpp
$ mkdir build && cd build
$ cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_STATIC_LIBS=ON -DBUILD_SHARED_LIBS=OFF -DARCHIVE_INSTALL_DIR=. -G "Unix Makefiles" ..
$ make
$ make install
```

### 编译程序

执行`$ bash build.sh`编译程序，在./val中放置图片用于图片检测，创建文件夹./output_img用于存放输出图片, 项目的结构如下：

```bash
yolov8n_lite_arm_cpu
├── build/
├── output/
├── val/
├─ build.sh
├─ CMakeLists.txt
├─ config.json
├─ README.md
├─ v8deploy_armopt.nb
└─ yolov8n_lite_arm_cpu.cc
```

### 运行

```bash
./yolov8_lite_arm_cpu image_test val/ # 从val读取10张图片进行检测, 检测后的图片保存在output中
./yolov8_lite_arm_cpu video_test test_video.mp4 # 读取test_video.mp4并将输出的视频保存在output中
./yolov8_lite_arm_cpu /dev/video* #使用/dev/video10摄像头进行检测，在命令行中显示检测参数，并在ouput中将当前检测帧保存为Camfram_yolov8n_lite_falldetect.jpg
```

### 配置模型config.json

```json
{
    "confidence_thres": 0.70,
    "model_path": "yolov8_falldet_3cls_arm_opt.nb",
    "SERVER_ADDRESS": "tcp://8.134.150.174:1883",
    "mqtt_topic": "820_cmd",
    "location": "apartment:1 floor:2 room:3"
}
```

### Q&A

如果出现摄像头：

```bash
[ERROR]Could not open camera
```

检查摄像头/dev/video*代号, 默认使用/dev/video10, 根据实际情况修改:

```bash
v4l2-ctl --list-device #如果出现Permission denied 需要sudo权限
# LRCP  USB2.0 (usb-xhci-hcd.9.auto-1):
#         /dev/video10
#         /dev/video11
```

出现Permission denied为摄像头配置可执行权限：

```bash
sudo chmod 777 /dev/video10
```
