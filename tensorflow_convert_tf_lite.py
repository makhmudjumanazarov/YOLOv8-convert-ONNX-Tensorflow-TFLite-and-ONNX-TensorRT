import tensorflow as tf

saved_model_dir = '/home/airi/yolo/YOLO-convert-TFLite-Tensorflow-TensorRT-ONNX/models/yolov8.tf'
tflite_model_path = '/home/airi/yolo/YOLO-convert-TFLite-Tensorflow-TensorRT-ONNX/models/yolov8.tflite'

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()

# Save the model
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)