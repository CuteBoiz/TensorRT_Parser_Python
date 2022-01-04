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
	assert args.batch_size > 0, 'Batch size must be greater than 0'
	assert os.path.isfile(args.weight), 'Could not found {}'.format(args.weight)
	assert os.path.isdir(args.data), 'Could not found {}'.format(args.data)

	filesName = glob.glob(os.path.join(args.data ,"*"))
	nrof_images = 0
	engine = TRTInference(trt_engine_path=args.weight)

	for file in filesName:
		file_name, file_extension = os.path.splitext(file)
		if file_extension == ".png" or file_extension == ".jpg" or file_extension == ".jpeg" or file_extension == ".bmp":
			print(os.path.basename(file))
			image = cv2.imread(file)
			nrof_images += 1
			start = time.time()
			results = engine.infer(image)
			for result in results:
				result = np.squeeze(result)
				if (args.softmax):
					result = softmax(result)
				print(result)
			end = time.time()
			print("{0:.0f}ms".format((end - start)*1000))
	print("Total inferenced images: {}".format(nrof_images))

def export_main(args):
	assert args.max_batch_size > 0, 'Max batch size must be greater than 0'
	assert os.path.isfile(args.weight), 'Could not found {}'.format(args.weight)

	if args.saved_name is not None:	
		if os.path.dirname(args.saved_name) != '':
			os.makedirs(os.path.dirname(args.saved_name), exist_ok=True)
		saved_path = args.saved_name
	else:
		dirname = os.path.dirname(args.weight)
		filename = os.path.basename(args.weight)
		saved_path = dirname + filename.replace(os.path.splitext(filename)[-1], '.trt')

	if args.dim is not None:
		assert args.input_tensor_name is not None
		exportTRTEngine(args.weight, saved_path, args.max_batch_size, args.input_tensor_name, args.dim, True, args.fp16)
	else:
		exportTRTEngine(onnx_file_name=args.weight, trt_file_name=saved_path, max_batch_size=args.max_batch_size, multi_dimension=False, FP16_MODE=args.fp16)

if __name__ == '__main__':
	parser = argparse.ArgumentParser('Export TensorRT')
	subparser = parser.add_subparsers(dest='mode')

	infer_parser = subparser.add_parser("infer")
	infer_parser.add_argument("--weight", type=str, required=True, help="TensorRT engine")
	infer_parser.add_argument("--data", type=str, required=True, help="Image folder path.")
	infer_parser.add_argument("-bs", "--batch_size", type=int, default=1, help="Infer batch size")
	infer_parser.add_argument("--softmax", action='store_true', default=False, help="Use softmax")
	
	export_parser = subparser.add_parser("export")
	export_parser.add_argument('--weight', type=str, required=True, help='Input model path')
	export_parser.add_argument('--saved_name', type=str, default=None, help='Output file name')
	export_parser.add_argument('--max_batch_size', type=int, default=1, help='max_batch_size')

	export_parser.add_argument('--dim', action='store', dest='dim', type=int, nargs='*', default=None, help='CHW or HWC size')
	export_parser.add_argument('--input_tensor_name', type=str, default=None, help='Input tensor name')
	export_parser.add_argument('--fp16', action='store_true', help='FP16 Convert')
	
	args = parser.parse_args()

	if args.mode == "infer":
		infer_main(args)
	if args.mode == "export":
		export_main(args)

