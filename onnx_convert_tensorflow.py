import onnx_tf 
import onnx

# Load  ONNX model
onnx_model = onnx.load('/home/airi/yolo/YOLO-convert-TFLite-Tensorflow-TensorRT-ONNX/models/yolov8.onnx')

# Convert ONNX model to TensorFlow format
tf_model = onnx_tf.backend.prepare(onnx_model)

# Export  TensorFlow  model 
tf_model.export_graph("/home/airi/yolo/YOLO-convert-TFLite-Tensorflow-TensorRT-ONNX/models/yolov8.tf")