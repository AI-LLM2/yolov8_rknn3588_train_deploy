#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/11/20 12:40
# @Author  : rzh
# @Site    : 
# @File    : test_seg.py
# @Software: PyCharm

#predict
from ultralytics import YOLO
import torch
import cv2
import os,datetime
# os.environ['KMP_DUPLICATE_LIB_OK']='True'    #如果训练时候报Initializing libiomp5md.dll, but found libiomp5md.dll already initialized错误，加上这句
# model = YOLO('yolov8n-pose.pt') #已经训练好的模型
model = YOLO('yolov8n-seg.pt') #已经训练好的模型
# model = YOLO('yolov8n.pt') #已经训练好的模型
# model = YOLO('yolov8x.pt') #已经训练好的模型
# model = YOLO('zhiguang_total.pt') #已经训练好的模型
# model = YOLO('./weights/yolov8_relu.pt')
# model = YOLO('./ultralytics/cfg/models/v8/yolov8n.yaml')
# model = YOLO('./weights/yolov8_relu_dict.pt.pt')
# model = YOLO('runs/segment/train31/weights/best.pt') #已经训练好的模型
# Define path to the image file

## Run inference on the source
# source = '/F_Pan/rknn/git/rknn_yolov8_cpp/tests' #待预测的数据保存路径
# results = model(task='detect', source=source, mode='predict', line_thickness=3, show=True, save=True, device='cpu')  # list of Results objects
# results = model(task='pose', source=source, mode='predict', line_thickness=3, show=True, save=True, device='cpu')  # list of Results objects

## export model onnx
# 指定输入和输出导出onnx
# input_names = [ "input_1"]
# output_names = [ "output1" ]
# torch.onnx.export(model, (dummy_input1, dummy_input2), "name.onnx", verbose=True, input_names=input_names, output_names=output_names)

print("===========  onnx =========== ")
model.export(format='onnx',opset=12)
print("======================== convert onnx Finished! .... ")
