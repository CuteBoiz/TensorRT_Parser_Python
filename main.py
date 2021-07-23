from TRT_Eng import TRTInference, exportTRTEngine
import cv2
import numpy as np
import argparse
import glob
import os
import time

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def infer_main(args):
	if args.batch_size <= 0:
		err_str = 'Batch size must be greater than 0'
		raise ValueError(err_str)
	if not os.path.isfile(args.weight):
		err_str = 'Could not found {}'.format(args.weight)
		raise ValueError(err_str)
	if not os.path.isdir(args.path):
		err_str = 'Could not found {}'.format(args.path)
		raise ValueError(err_str)

	filesName = glob.glob(os.path.join(args.path ,"*"))
	nrof_images = 0
	engine = TRTInference(trt_engine_path=args.weight)

	for file in filesName:
		file_name, file_extension = os.path.splitext(file)
		if file_extension == ".png" or file_extension == ".jpg" or file_extension == ".jpeg" or file_extension == ".bmp":
			print(os.path.basename(file))
			image = cv2.imread(file)
			nrof_images += 1
			start = time.time()
			results = engine.infer(images)
			for result in results:
				result = np.squeeze(result)
				if (args.softmax):
					result = softmax(result)
				print(result)
			end = time.time()
			print("{0:.0f}ms".format((end - start)*1000))
	print("Total inferenced images: {}".format(nrof_images))


def export_main(args):
	if args.max_batch_size <= 0:
		err_str = 'Max batch size must be greater than 0'
		raise ValueError(err_str)
	if not os.path.isfile(args.weight):
		err_str = 'Could not found {}'.format(args.weight)
		raise ValueError(err_str)
	basename = args.name.split("/")[-1]
	if basename.rfind('.') == -1:
		err_str = "{} is an unvalid filename".format(basename)
		raise ValueError(err_str)
	if os.path.dirname(args.name) != '':
		os.makedirs(os.path.dirname(args.name), exist_ok=True)

	if args.ds:
		assert args.dimension is not None
		assert len(args.dimension) == 3
		assert args.input_tensor_name is not None
		exportTRTEngine(args.weight, args.name, args.max_batch_size, args.input_tensor_name, args.dimension, True, args.fp16)
	else:
		exportTRTEngine(onnx_file_name=args.weight, trt_file_name=args.name, max_batch_size=args.max_batch_size, multi_dimension=False, FP16_MODE=args.fp16)

if __name__ == '__main__':
	parser = argparse.ArgumentParser('Export TensorRT')
	subparser = parser.add_subparsers(dest='mode')

	infer_parser = subparser.add_parser("infer")
	infer_parser.add_argument("--weight", type=str, required=True, help="TensorRT engine")
	infer_parser.add_argument("--path", type=str, required=True, help="Image folder path.")
	infer_parser.add_argument("-bs", "--batch_size", type=int, default=1, help="Infer batch size")
	infer_parser.add_argument("--softmax", action='store_true', default=False, help="Use softmax")
	
	export_parser = subparser.add_parser("export")
	export_parser.add_argument('--weight', type=str, required=True, help='Input model path')
	export_parser.add_argument('--name', type=str, required=True, help='Output file name')
	export_parser.add_argument('--max_batch_size', type=int, default=1, help='max_batch_size')

	export_parser.add_argument('--ds', action='store_true', help='Dynamic Shape Convert')
	export_parser.add_argument('--dimension', action='store', dest='dimension', type=int, nargs='*', default=None, help='CHW or HWC size')
	export_parser.add_argument('--input_tensor_name', type=str, default=None, help='Input tensor name')
	export_parser.add_argument('--fp16', action='store_true', help='FP16 Convert')
	
	args = parser.parse_args()

	if args.mode == "infer":
		infer_main(args)
	if args.mode == "export":
		export_main(args)

