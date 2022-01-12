
from numpy.lib.function_base import append
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import numpy as np
import threading
import time
import cv2


def show_engine_info(engine):
	assert engine is not None
	
	print("[INFO] TensorRT Engine Info")
	print(f"\t + Max batch size: {engine.max_batch_size}.")
	print(f"\t + Engine mem size: {engine.device_memory_size/(1048576)} MB (GPU Mem).")
	print("\t + Tensors:")
	for binding in engine:
		if engine.binding_is_input(binding):
			print(f"\t\t + Input: ", end='')
		else:
			print(f"\t\t + Output: ", end='')
		print(engine.get_binding_shape(binding))
	

class TRTInference():
	def __init__(self, engine_path, gpu_num=0 ,trt_engine_datatype=trt.DataType.FLOAT):
		self.cfx = cuda.Device(gpu_num).make_context()
		stream = cuda.Stream()

		TRT_LOGGER = trt.Logger(trt.Logger.INFO)
		trt.init_libnvinfer_plugins(TRT_LOGGER, '')
		runtime = trt.Runtime(TRT_LOGGER)

		# Deserialize engine
		with open(engine_path, 'rb') as f:
			buf = f.read()
			engine = runtime.deserialize_cuda_engine(buf)
		show_engine_info(engine)
		context = engine.create_execution_context()

		# Get input shape
		dimension = engine.get_binding_shape(engine[0])
		if dimension[1] == 3 or dimension[1] == 1:
			self.channel_first = True
			self.input_channels = dimension[1]
			self.input_height = dimension[2]
			self.input_width = dimension[3]
		if dimension[3] == 3 or dimension[3] == 1:
			self.channel_first = False
			self.input_channels = dimension[3]
			self.input_height = dimension[1]
			self.input_width = dimension[2]

		self.context = context
		self.engine  = engine
		self.stream = stream

	
	def preprocess_images(self, images):
		assert isinstance(images, list)
		x = []
		for image in images:
			assert image is not None
			image = cv2.resize(src=image, dsize=(self.input_width, self.input_height), interpolation = cv2.INTER_AREA)
			image = np.float32(image)
			image = image*(1/255)
			mean = [0.485, 0.456, 0.406]
			std = [0.229, 0.224, 0.225]
			image = (image - mean) / std
			if self.channel_first:
				image = image.transpose((2, 0, 1))
			x.append(image)
		x = np.asarray(x).astype(np.float32)
		return x

	def infer(self, images):
		assert len(images) <= self.engine.max_batch_size, f"[ERROR] Batch size num must be smaller than {self.engine.max_batch_size}"
		threading.Thread.__init__(self)
		# Image preprocessing
		x = self.preprocess_images(images)
	
		# Create buffers & Allocate images
		bindings = []
		host_inputs  = []
		host_outputs = []
		device_inputs  = []
		device_outputs = []
		output_shape = []

		for binding in self.engine:
			size = trt.volume(self.engine.get_binding_shape(binding))
			dtype = trt.nptype(self.engine.get_binding_dtype(binding))
			host_mem = cuda.pagelocked_empty(size, dtype)
			device_mem = cuda.mem_alloc(host_mem.nbytes)
			bindings.append(int(device_mem))

			if self.engine.binding_is_input(binding):
				host_inputs.append(host_mem)
				device_inputs.append(device_mem)
			else:
				output_shape.append(self.engine.get_binding_shape(binding))
				host_outputs.append(host_mem)
				device_outputs.append(device_mem)

		self.cfx.push()
		host_inputs[0] = np.ascontiguousarray(x)
		
		# Inference
		outputs = []
		cuda.memcpy_htod_async(device_inputs[0], host_inputs[0], self.stream)
		self.context.execute_async(batch_size=len(images) ,bindings=bindings, stream_handle=self.stream.handle)

		for i in range(len(host_outputs)):
			cuda.memcpy_dtoh_async(host_outputs[i], device_outputs[i], self.stream)
			outputs.append(host_outputs[i].reshape(output_shape[i]))
		self.stream.synchronize()
		self.cfx.pop()

		result = []
		for output in outputs:
			result.append(output[:len(images),:])
		return result

	def __del__(self):
		self.cfx.pop()

def exportTRTEngine(onnx_file_name, trt_file_name, max_batch_size, max_workspace_size, input_tensor_name=None, dimension=None, FP16_MODE=False):
	assert onnx_file_name is not None
	assert trt_file_name is not None
	assert max_batch_size is not None
	assert max_workspace_size is not None

	TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
	TRT_VERSION_MAJOR = int(trt.__version__.split('.')[0])

	with trt.Builder(TRT_LOGGER) as builder:
		flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
		network = builder.create_network(flag)
		parser = trt.OnnxParser(network, TRT_LOGGER)
		success = parser.parse_from_file(onnx_file_name)
		for idx in range(parser.num_errors):
			print(parser.get_error(idx))
		if not success:
			 raise RuntimeError("Onnx model parsing from {} failed. Error: {}".format(onnx_file_name, parser.get_error(0).desc()))
		
		builder.max_batch_size = max_batch_size
		config = builder.create_builder_config()

		if TRT_VERSION_MAJOR == 7:
			builder.max_workspace_size = 1048576 * max_workspace_size
		elif TRT_VERSION_MAJOR == 8:
			config.max_workspace_size = 1048576 * max_workspace_size

		if not FP16_MODE:
			print('Converting into FP32 (default), max_batch_size={}'.format(max_batch_size))
		else:
			if not builder.platform_has_fast_fp16:
				print('Warning: This platform is not optimized for fp16 fast mode')
			else:
				if TRT_VERSION_MAJOR == 7:
					builder.fp16_mode = True
				elif TRT_VERSION_MAJOR == 8:
					config.set_flag(trt.BuilderFlag.FP16)
				print('Converting into FP16, max_batch_size={}'.format(max_batch_size))	

		if dimension is not None:
			profile = builder.create_optimization_profile()
			profile.set_shape(input_tensor_name, (1, dimension[0], dimension[1], dimension[2]), (max_batch_size, dimension[0], dimension[1], dimension[2]), (max_batch_size, dimension[0], dimension[1], dimension[2]))
			config.add_optimization_profile(profile)

		enigne = builder.build_engine(network, config)

		if TRT_VERSION_MAJOR == 7:
			serialized_engine = enigne.serialize()
		elif TRT_VERSION_MAJOR == 8:
			serialized_engine = builder.build_serialized_network(network, config)

		with open(trt_file_name, "wb") as trt_model_file:
			trt_model_file.write(serialized_engine)

		show_engine_info(enigne)
		print("Export Done!!")