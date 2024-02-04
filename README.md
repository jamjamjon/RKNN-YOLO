#  RKNN-YOLO CPP代码部署 (RK3399Pro)

提供anchor-free和anchor-based的c++参考代码
- YOLOv5-Style(anchor-based)
- YOLOX-Style(anchor-free)

### 已支持
- [x] detection 目标检测
- [x] tracking 多目标跟踪
- [x] keypoints detection (pose-estimation) 关键点检测
- [x] yolov5-Style/data/weights提供2个rknn模型：human.rknn, pose.rknn
- [x] yoloX-Style/data/weights提供1个coco.rknn模型


# Installation
-  install `eigen`
- install`opencv`
- compile：
```bash
mkdir build
cd build
cmake ..
make
```
	

# 快速开始
## 目标检测(detection):

1. YOLOX-Style
```bash
./main ../data/weights/test.rknn ../data/images/bus.jpg .4 .4
```

2. YOLOv5-Styl
```bash
./main 0 ../data/weights/test.rknn ../data/images/bus.jpg .4 .4
```
	
	

## 目标跟踪(tracking): 
**src/main.cpp下，取消Track注释即可**
```cpp
auto start = chrono::system_clock::now(); // start
model->Detect(frame, CONF_THRESH, NMS_THRESH); // detect
// model->Track(tracker, frame, false); // tracking(enable_trajectory=false)
auto end = chrono::system_clock::now();  // end
```
	
## 关键点检测(pose):

当前仅提供了YOLOv5-Style关键点decode代码，并且提供量化转换后的RKNN模型 => `YOLOv5/Stle/data/weights/pose.rknn`。
由于8bit量化,关键点偏移较大，建议使用FP16模型。

YOLOv5-Style 
```bash
./main 2 ../data/weights/test.rknn ../data/images/bus.jpg .4 .4
```	


## rknn模型导出
pipeline: `pytorch => onnx => rknn`
可以从本人另一个repo导出：(https://github.com/jamjamjon/YOLOXY/tree/master)
```bash
python tool.py --include rknn
```
yolov5的导出可以参考YOLOXY中`common.py` => `DetectX()`。


# 注意
`YOLOX-Style/data/weights`下的`.rknn`模型仅供测试。
