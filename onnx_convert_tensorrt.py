'''
    Target: TensorRT conversion
    model: Resnet 18
    Image Resolution: [1, 3, 640, 640]
    Tensorrt: 8.6.1
    Python3: OOP Methodology
'''

import tensorrt as trt


class TensorRTConversion:
    '''
        path : to onnx
        path : to engine
        maxworkspace : gb < 1gb
        precision : 16 float and half precision 
        Inference mode: Dynamic Batch [1, 10, 20] 
    '''

    def __init__(self, path_to_onnx, path_to_engine, max_workspace_size=6 << 30, half_precision=False):
        
        self.TRT_LOGGER = trt.Logger(trt.Logger.WARNING)  # To warning errors
        self.path_to_onnx = path_to_onnx
        self.path_to_engine = path_to_engine
        self.max_workspace_size = max_workspace_size
        self.half_precision = half_precision
    
    '''
        {
            INIT BUILD
            INIT CONFIG
            INIT EXPLICIT BATCH
            INIT NETWORK 
        }
        Tensorrt >= 8.0.0
    '''
    def convert(self):
        builder = trt.Builder(self.TRT_LOGGER)
        config = builder.create_builder_config()

        config.max_workspace_size = self.max_workspace_size
        explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(explicit_batch)
        parser = trt.OnnxParser(network, self.TRT_LOGGER)

        with open(self.path_to_onnx, 'rb') as model_onnx:
            if not parser.parse(model_onnx.read()):
                print('ERROR: Failed to parse Onnx Model')
                for error in parser.errors:
                    print(error)
                return None
        # # Set profile for explicit batch
        # profile = builder.create_optimization_profile()
        # profile.set_shape('input_name', min= (1, 3, 224, 224), opt=(10, 3, 640, 640), max=(20, 3, 640, 640))
        # config.add_optimization(profile)
        # print('Successfully TensorRT Engine Configured to Max Batch ')
        # print('\n') 

        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        engine = builder.build_engine(network, config)

        with open(self.path_to_engine, "wb") as f_engine:
            f_engine.write(engine.serialize())
        
        print("Successfully Converted ONNX to Tensorrt Dynamic Engine")
        print(f'Serialized engine saved in engine path: {self.path_to_engine}')

convert = TensorRTConversion('/home/airi/yolo/YOLO-convert-TFLite-Tensorflow-TensorRT-ONNX/models/yolov8.onnx', '/home/airi/yolo/YOLO-convert-TFLite-Tensorflow-TensorRT-ONNX/models/yolov8_1.engine')
convert.convert()