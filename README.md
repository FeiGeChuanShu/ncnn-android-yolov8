# ncnn-yolov8 detection and segmentation demo

The yolov8 object detection and segmentation

1.Android yolov8 detection demo  
2.yolov8s-seg.cpp

## convert to onnx for ncnn
1.change c2f split to slice  
![](./doc/c2f.jpg)  
2.for Detection model change class Detect output  
![](./doc/Detect.jpg)  
3.for Segmentation model change class Detect output and Segment output  
![](./doc/Detect-seg.jpg)  
![](./doc/Segment.jpg)  
## screenshot
![](./ncnn-android-yolov8/screenshot.png)
![](yolov8s-seg.jpg)
## Reference：  
https://github.com/nihui/ncnn-android-nanodet  
https://github.com/Tencent/ncnn  
https://github.com/ultralytics/assets/releases/tag/v0.0.0
