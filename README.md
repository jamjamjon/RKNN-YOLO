#  RKNN-YOLO CPP代码部署 (RK3399Pro)

### 提供anchor-free和anchor-based的c++参考代码
YOLOv5-Style(anchor-based)
YOLOX-Style(anchor-free)

### 已支持
[x] detection 目标检测\
[x] tracking 多目标跟踪\
[x] keypoints detection (pose-estimation) 关键点检测\
[x] yolov5-Style/data/weights提供2个rknn模型：human.rknn, pose.rknn\
[x] yoloX-Style/data/weights提供1个coco.rknn模型



### 准备工作
1. 安装eigen.zip; 安装opencv
2. 修改cmakelist，编译：

	mkdir build\
	cd build\
	cmake ..\
	make


### 目标检测(detection):

1.YOLOX-Style

	./main ../data/weights/test.rknn ../data/images/bus.jpg .4 .4


2.YOLOv5-Styl

	./main 0 ../data/weights/test.rknn ../data/images/bus.jpg .4 .4
	

### 目标跟踪(tracking): 
	src/main.cpp下，取消Track注释即可

	auto start = chrono::system_clock::now();  			// start time
	model->Detect(frame, CONF_THRESH, NMS_THRESH);		// detect
	// model->Track(tracker, frame, false);					// tracking	enable_trajectory=true
	auto end = chrono::system_clock::now();   // end time



### 关键点检测(pose):

当前仅在YOLOv5-Style下提供了关键点decode代码，并且提供量化转换后的RKNN模型(YOLOv5/Stle/data/weights/pose.rknn)，量化后的rknn模型在关键点检测上表现很差，关键点偏移太大，decode关键点代码可参考该repo，模型方面建议参考RK官方混合量化，本人未尝试。

YOLOv5-Style 
	
	./main 2 ../data/weights/test.rknn ../data/images/bus.jpg .4 .4



## 注意
YOLOX-Style/data/weights下的.rknn模型是测试模型，该模型还没训练完毕，仅仅作测试使用，后续会更新该rknn模型。
后续仅仅对YOLOX-Style的代码进行优化，YOLOv5的不再进行维护。

## 关于rknn模型导出
pytorch ==> onnx ==> rknn\
可以从本人另一个repo导出：YOLOXY。\
yolov5的导出可以参考YOLOXY中common.py ==> DetectX()。



### 后续完善readme...
