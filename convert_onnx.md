# setup yolov5 repo 
- download repo 
```
    git clone https://github.com/ultralytics/yolov5.git
```
- setup docker 
```
    nvidia-docker run --name yolor -it -v /mnt/ssd1/thang/yolor:/workspace/share -v /mnt/ssd1/thang/coco_stuff_yolo:/workspace/share/coco_stuff_yolo -v /data1/thang/datasets:/workspace/share/coco --shm-size=64g -p 6000:6000 -p 6001:6001 nvcr.io/nvidia/pytorch:21.08-py3
```
# train yolov5 model 
- download model from original [repo](https://github.com/ultralytics/yolov5)  
- convert coco format dataset into yolo format using  [generate_data.py](generate_data.py) file

# export onnx model
- setup yolov5-rt-stack repo 
```
https://github.com/thangnx183/yolov5-rt-stack.git
cd yolov5-rt-stack
pip install -e .
```
- using [demo.py](demo.py) to export onnx