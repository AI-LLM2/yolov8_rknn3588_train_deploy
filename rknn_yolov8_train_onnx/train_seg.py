#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/11/18 9:33
# @Author  : rzh
# @Site    : 
# @File    : train_seg.py
# @Software: PyCharm

# train model
from ultralytics import YOLO
# Load a model 三选一
model = YOLO('./ultralytics/cfg/models/v8/yolov8-seg.yaml')  # build a new model from YAML 配置好模型和训练数据yaml文件信息
# model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
# model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights
# model = YOLO('yolov8n-seg.pt')

# Train the model
# model.train(data='./ultralytics/cfg/datasets/coco128-seg.yaml', epochs=100, imgsz=640)
# model.train(data='./ultralytics/cfg/datasets/coco128-seg.yaml', epochs = 10)
model.train(data='./ultralytics/cfg/datasets/superai-seg.yaml', epochs = 100)

# model('https://ultralytics.com/images/bus.jpg')
model('bus.jpg')