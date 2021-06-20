
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import numpy as np
import threading
import time
import pycuda.autoinit
from pycuda.driver import Context
import cv2

class TRTInference():
    def __init__(self, trt_engine_path, trt_engine_datatype = trt.DataType.FLOAT, batch_size = 1):
        self.cfx = cuda.Device(0).make_context()
        stream = cuda.Stream()

        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(TRT_LOGGER, '')
        runtime = trt.Runtime(TRT_LOGGER)

        # deserialize engine
        with open(trt_engine_path, 'rb') as f:
            buf = f.read()
            engine = runtime.deserialize_cuda_engine(buf)
        context = engine.create_execution_context()

        # prepare buffer
        host_inputs  = []
        cuda_inputs  = []
        host_outputs = []
        cuda_outputs = []
        bindings = []
        output_shape = []

        dimension = engine.get_binding_shape(engine[0])
        if dimension[1] == 3: #channel first
            self.input_height = dimension[2]
            self.input_width = dimension[3]
        if dimension[3] == 3: #channel last
            self.input_height = dimension[1]
            self.input_width = dimension[2]

        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            host_mem = cuda.pagelocked_empty(size, np.float32)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)

            bindings.append(int(cuda_mem))
            if engine.binding_is_input(binding):
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                output_shape.append(engine.get_binding_shape(binding))
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)

        # store
        self.stream  = stream
        self.context = context
        self.engine  = engine

        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings
        self.output_shape = output_shape
    

    def infer(self, images):
        threading.Thread.__init__(self)
        for image in images:
            cv2.resize(src=image, dsize=(self.input_width, self.input_height), dst=image, interpolation = cv2.INTER_AREA)
        self.cfx.push()

        # read image
        np.copyto(self.host_inputs[0], image.ravel())
        output = []
        # inference
        cuda.memcpy_htod_async(self.cuda_inputs[0], self.host_inputs[0], self.stream)
        self.context.execute_async(bindings=self.bindings, stream_handle=self.stream.handle)
        for index, cuda_output in enumerate(self.cuda_outputs):
            cuda.memcpy_dtoh_async(self.host_outputs[index], cuda_output, self.stream)
            output.append(self.host_outputs[index].reshape(self.output_shape[index]))
        self.stream.synchronize()
        self.cfx.pop()
        return output

    def destory(self):
        self.cfx.pop()


def exportTRTEngine(onnx_file_name, trt_file_name, max_batch_size, input_tensor_name="", dimension="", multi_dimension=False, FP16_MODE=False):
    assert onnx_file_name is not None
    assert trt_file_name is not None
    assert max_batch_size is not None

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    TRT_VERSION_MAJOR = int(trt.__version__.split('.')[0])

    with trt.Builder(TRT_LOGGER) as builder:
        flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        network = builder.create_network(flag)
        parser = trt.OnnxParser(network, TRT_LOGGER)

        if not FP16_MODE:
            print('Converting into FP32 (default), max_batch_size={}'.format(max_batch_size))
            builder.fp16_mode = False
        else:
            if not builder.platform_has_fast_fp16:
                print('Warning: This platform is not optimized for fast fp16 mode')
            builder.fp16_mode = True
            print('Converting into FP16, max_batch_size={}'.format(max_batch_size))

        builder.max_workspace_size = 1 << 30
        builder.max_batch_size = max_batch_size

        with open(onnx_file_name, 'rb') as onnx_model_file:
            onnx_model = onnx_model_file.read()

        if not parser.parse(onnx_model):
            raise RuntimeError("Onnx model parsing from {} failed. Error: {}".format(onnx_file_name, parser.get_error(0).desc()))

        config = builder.create_builder_config()

        if multi_dimension == True:
            profile = builder.create_optimization_profile()
            profile.set_shape(input_tensor_name, (1, dimension[0], dimension[1], dimension[2]), (max_batch_size, dimension[0], dimension[1], dimension[2]), (max_batch_size, dimension[0], dimension[1], dimension[2]))
            config.add_optimization_profile(profile)

        trt_model= builder.build_engine(network, config)

        serialized_trt_model = trt_model.serialize()
        with open(trt_file_name, "wb") as trt_model_file:
            trt_model_file.write(serialized_trt_model)
        print("Export Done!!")