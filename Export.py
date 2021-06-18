import tensorrt as trt 
import common
import argparse

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

if __name__ == '__main__':
	parser = argparse.ArgumentParser('Export TensorRT')
	parser.add_argument('--weight', type=str, required=True, help='Input model path')
	parser.add_argument('--output', type=str, required=True, help='Output file name')
	parser.add_argument('--max_batch_size', type=int, required=True, help='max_batch_size')
	parser.add_argument('--m', action='store_true', help='Output file name')
	parser.add_argument('--dimension', action='store', dest='dimension', type=int, nargs='*', default=None, help='CHW or HWC size')
	parser.add_argument('--input_tensor_name', type=str, default=None, help='Input tensor name')
	parser.add_argument('--fp16', action='store_true', help='FP16 Convert')
	args=parser.parse_args()

	if args.m == True:
		assert args.dimension is not None
		assert len(args.dimension) == 3
		assert args.input_tensor_name is not None
		exportTRTEngine(args.weight, args.output, args.max_batch_size, args.input_tensor_name, args.dimension, True, args.fp16)
	else:
		exportTRTEngine(onnx_file_name=args.weight, trt_file_name=args.output, max_batch_size=args.max_batch_size, multi_dimension=False, FP16_MODE=args.fp16)