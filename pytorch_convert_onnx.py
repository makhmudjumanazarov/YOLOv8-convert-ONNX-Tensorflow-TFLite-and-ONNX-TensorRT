# import torch
# import numpy as np
# import cv2
# import pandas as pd 
# import onnx

# from matplotlib import pyplot as plt

# PATH = "/home/airi/yolo/YOLO-convert-TFLite-Tensorflow-TensorRT-ONNX/models/yolov8.pt"


# # How to convert pytorch to onnx format
# device = torch.device('cpu')
# model = torch.load(PATH, map_location=device)['model'].float()
# torch.onnx.export(model, torch.zeros((1, 3, 640, 640)), '/home/airi/yolo/YOLO-convert-TFLite-Tensorflow-TensorRT-ONNX/models/yolov8.onnx', opset_version=12)

# onnx_model = onnx.load("/home/airi/yolo/YOLO-convert-TFLite-Tensorflow-TensorRT-ONNX/models/yolov8.onnx")
# onnx.checker.check_model(onnx_model)   


import torch

img_size = (640, 640)
batch_size = 1
onnx_model_path = '/home/airi/yolo/YOLO-convert-TFLite-Tensorflow-TensorRT-ONNX/models/yolov8.onnx'
PATH = "/home/airi/yolo/YOLO-convert-TFLite-Tensorflow-TensorRT-ONNX/models/yolov8.pt"

# How to convert pytorch to onnx format
# device = torch.device('cpu')
model = torch.load(PATH)['model'].float()

model.eval()

sample_input = torch.rand((batch_size, 3, *img_size))

y = model(sample_input)

torch.onnx.export(
    model,
    sample_input, 
    onnx_model_path,
    verbose=False,
    input_names=['input'],
    output_names=['output'],
    opset_version=12
)

