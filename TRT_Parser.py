
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

		batched_result = (outputs[3][:len(images),:])
		
		results = []
		for result in batched_result:
			boxes = self.non_max_suppression(result, origin_h=1700, origin_w=1700, conf_thres=0.2, nms_thres=0.45)
			result_boxes = boxes[:, :4] if len(boxes) else np.array([])
			result_scores = boxes[:, 4] if len(boxes) else np.array([])
			result_classid = boxes[:, 5] if len(boxes) else np.array([])
			results.append((result_boxes, result_scores, result_classid))

		return results

	def xywh2xyxy(self, origin_h, origin_w, x):
		"""
		description:    Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
		param:
			origin_h:   height of original image
			origin_w:   width of original image
			x:          A boxes numpy, each row is a box [center_x, center_y, w, h]
		return:
			y:          A boxes numpy, each row is a box [x1, y1, x2, y2]
		"""
		y = np.zeros_like(x)
		r_w = self.input_width / origin_w
		r_h = self.input_height / origin_h
		if r_h > r_w:
			y[:, 0] = x[:, 0] - x[:, 2] / 2
			y[:, 2] = x[:, 0] + x[:, 2] / 2
			y[:, 1] = x[:, 1] - x[:, 3] / 2 - (self.input_height - r_w * origin_h) / 2
			y[:, 3] = x[:, 1] + x[:, 3] / 2 - (self.input_height - r_w * origin_h) / 2
			y /= r_w
		else:
			y[:, 0] = x[:, 0] - x[:, 2] / 2 - (self.input_width - r_h * origin_w) / 2
			y[:, 2] = x[:, 0] + x[:, 2] / 2 - (self.input_width - r_h * origin_w) / 2
			y[:, 1] = x[:, 1] - x[:, 3] / 2
			y[:, 3] = x[:, 1] + x[:, 3] / 2
			y /= r_h

		return y

	def bbox_iou(self, box1, box2, x1y1x2y2=True):
		"""
		description: compute the IoU of two bounding boxes
		param:
			box1: A box coordinate (can be (x1, y1, x2, y2) or (x, y, w, h))
			box2: A box coordinate (can be (x1, y1, x2, y2) or (x, y, w, h))            
			x1y1x2y2: select the coordinate format
		return:
			iou: computed iou
		"""
		if not x1y1x2y2:
			# Transform from center and width to exact coordinates
			b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
			b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
			b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
			b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
		else:
			# Get the coordinates of bounding boxes
			b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
			b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

		# Get the coordinates of the intersection rectangle
		inter_rect_x1 = np.maximum(b1_x1, b2_x1)
		inter_rect_y1 = np.maximum(b1_y1, b2_y1)
		inter_rect_x2 = np.minimum(b1_x2, b2_x2)
		inter_rect_y2 = np.minimum(b1_y2, b2_y2)
		# Intersection area
		inter_area = np.clip(inter_rect_x2 - inter_rect_x1 + 1, 0, None) * \
						np.clip(inter_rect_y2 - inter_rect_y1 + 1, 0, None)
		# Union Area
		b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
		b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

		iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

		return iou


	def non_max_suppression(self, prediction, origin_h, origin_w, conf_thres=0.5, nms_thres=0.4):
		"""
		description: Removes detections with lower object confidence score than 'conf_thres' and performs
		Non-Maximum Suppression to further filter detections.
		param:
			prediction: detections, (x1, y1, x2, y2, conf, cls_id)
			origin_h: original image height
			origin_w: original image width
			conf_thres: a confidence threshold to filter detections
			nms_thres: a iou threshold to filter detections
		return:
			boxes: output after nms with the shape (x1, y1, x2, y2, conf, cls_id)
		"""
		# Get the boxes that score > CONF_THRESH
		boxes = prediction[prediction[:, 4] >= conf_thres]
		# Trandform bbox from [center_x, center_y, w, h] to [x1, y1, x2, y2]
		boxes[:, :4] = self.xywh2xyxy(origin_h, origin_w, boxes[:, :4])
		# clip the coordinates
		boxes[:, 0] = np.clip(boxes[:, 0], 0, origin_w -1)
		boxes[:, 2] = np.clip(boxes[:, 2], 0, origin_w -1)
		boxes[:, 1] = np.clip(boxes[:, 1], 0, origin_h -1)
		boxes[:, 3] = np.clip(boxes[:, 3], 0, origin_h -1)
		# Object confidence
		confs = boxes[:, 4]
		# Sort by the confs
		boxes = boxes[np.argsort(-confs)]
		# Perform non-maximum suppression
		keep_boxes = []
		while boxes.shape[0]:
			large_overlap = self.bbox_iou(np.expand_dims(boxes[0, :4], 0), boxes[:, :4]) > nms_thres
			label_match = boxes[0, -1] == boxes[:, -1]
			# Indices of boxes with lower confidence scores, large IOUs and matching labels
			invalid = large_overlap & label_match
			keep_boxes += [boxes[0]]
			boxes = boxes[~invalid]
		boxes = np.stack(keep_boxes, 0) if len(keep_boxes) else np.array([])
		return boxes

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


	