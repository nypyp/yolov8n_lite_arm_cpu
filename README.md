## yolov8 fall detect
设备：Firfly RK3390J

### 安装PaddleLite库文件


[PaddleLite 预编译库v2.13-rc](https://github.com/PaddlePaddle/Paddle-Lite/releases/tag/v2.13-rc)
下载预编译库，选择armlinux.armv8.gcc.with_extra.with_cv版本

在demo/cxx路径下放置项目：
`./inference_lite_lib.armlinux.armv8/demo/cxx/yolov8n_lite_arm_cpu`

### 安装Paho Mqtt C++库

```bash
#递归克隆主仓库以及子仓库，即Paho mqtt c库，编译需要
$ git clone --recursive https://github.com/eclipse/paho.mqtt.cpp 
cd paho.mqtt.cpp

$ cmake -Bbuild -H. -DPAHO_WITH_MQTT_C=ON -DPAHO_BUILD_STATIC=ON \
-DPAHO_BUILD_SAMPLES=ON
$ sudo cmake --build build/ --target install
$ sudo ldconfig
```

### 编译程序
执行`$ bash build.sh`编译程序，在./val中放置图片用于图片检测，创建文件夹./output_img用于存放输出图片, 项目的结构如下：
```bash
yolov8n_lite_arm_cpu
├── build/
├── output_img/
├── val/
├─ build.sh
├─ CMakeLists.txt
├─ README.md
├─ v8deploy_armopt.nb
└─ yolov8n_lite_arm_cpu.cc
```

### 运行

```bash
./yolov8_lite_arm_cpu image_test val/ # 从val读取10张图片进行检测
./yolov8_lite_arm_cpu /dev/video* #使用/dev/video10摄像头进行检测
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

