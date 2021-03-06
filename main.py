'''
author: phatnt
date modified: 2021-01-12
'''

from TRT_Parser import TRTInference, exportTRTEngine
import cv2
import numpy as np
import argparse
import glob
import os
import time

def softmax(x):
	'''
		Softmax caculate
		Args:
			+ x <numpy array>
		Return:
			softmaxed x
	'''
	e_x = np.exp(x - np.max(x))
	return e_x / e_x.sum(axis=0)

def export(args):
	assert args.max_batch_size > 0, 'Max batch size must be greater than 0'
	assert os.path.isfile(args.weight), 'Could not found {}'.format(args.weight)

	# Get trt engine saved name
	if args.saved_name is not None:	
		if os.path.dirname(args.saved_name) != '':
			os.makedirs(os.path.dirname(args.saved_name), exist_ok=True)
		saved_path = args.saved_name
	else:
		dirname = os.path.dirname(args.weight)
		filename = os.path.basename(args.weight)
		saved_path = os.path.join(dirname, filename.replace(os.path.splitext(filename)[-1], '.trt'))

	# Export
	if args.dim is not None:
		assert args.input_tensor_name is not None
		exportTRTEngine(args.weight, saved_path, args.max_batch_size, args.max_workspace_size,args.input_tensor_name, args.dim, args.fp16)
	else:
		exportTRTEngine(onnx_file_name=args.weight, trt_file_name=saved_path, 
						max_batch_size=args.max_batch_size, max_workspace_size=args.max_workspace_size, FP16_MODE=args.fp16)

def infer(args):
	assert args.batch_size > 0, 'Batch size must be greater than 0'
	assert os.path.isfile(args.weight), 'Could not found {}'.format(args.weight)

	# Load data 
	images = []
	if os.path.isfile(args.data):
		extentions = args.data.split('.')[-1]
		if extentions in ['jpg', 'png', 'bmp', 'jpeg']:
			print(args.data)
			image = cv2.imread(args.data)
			images.append(image)
		elif extentions in ['mp4', 'mov', 'wmv', 'mkv', 'avi', 'flv']:
			cap = cv2.VideoCapture(args.data)
			if (cap.isOpened()== False):
  				print("Error opening video stream or file")
			while(cap.isOpened()):
				ret, frame = cap.read()
				if ret == True:
					images.append(frame)
				else: 
					break
	elif os.path.isdir(args.data):
		files = sorted(glob.glob(os.path.join(args.data, '*')))
		for file in files:
			extentions = file.split('.')[-1]
			if extentions in ['jpg', 'png', 'bmp', 'jpeg']:
				print(file)
				image = cv2.imread(file)
				images.append(image)
			else:
				continue
	else:
		raise Exception(f"Could not load any data from: {args.data}")

	# Batched
	batched_images = []
	range_num = len(images)//args.batch_size+1 if len(images)%args.batch_size > 0 else len(images)//args.batch_size
	for i in range(range_num):
		batched_images.append([])
	count  = 0
	index = 0
	for i in range(len(images)):
		batched_images[index].append(images[i])
		count+=1
		if count == args.batch_size:
			count = 0
			index += 1
	# Load engine
	engine = TRTInference(engine_path=args.weight, gpu_num=args.gpu)

	for batched in batched_images:
		start = time.time()
		results = engine.infer(batched)
		for result in results:
			result = np.squeeze(result)
			if (args.softmax):
				result = softmax(result)
			print(result.shape)
		end = time.time()
		print("{0:.0f}ms".format((end - start)*1000))
	print("Total inferenced images: {}".format(len(images)))

if __name__ == '__main__':
	parser = argparse.ArgumentParser('Export TensorRT')
	subparser = parser.add_subparsers(dest='mode')

	infer_parser = subparser.add_parser("infer")
	infer_parser.add_argument("--weight", type=str, required=True, help="TensorRT engine")
	infer_parser.add_argument("--data", type=str, required=True, help="Image folder path.")
	infer_parser.add_argument("-bs", "--batch_size", type=int, default=1, help="Infer batch size")
	infer_parser.add_argument("--softmax", action='store_true', default=False, help="Use softmax")
	infer_parser.add_argument("--gpu", type=int, default=0, help="Infer gpu num")
	
	
	export_parser = subparser.add_parser("export")
	export_parser.add_argument('--weight', type=str, required=True, help='Input model path')
	export_parser.add_argument('--saved_name', type=str, default=None, help='Output file name')
	export_parser.add_argument('--max_batch_size', type=int, default=1, help='max_batch_size')
	export_parser.add_argument('--max_workspace_size', type=int, default=1300, help='max workspace size(MB)')

	export_parser.add_argument('--dim', action='store', dest='dim', type=int, nargs='*', default=None, help='CHW or HWC size')
	export_parser.add_argument('--input_tensor_name', type=str, default=None, help='Input tensor name')
	export_parser.add_argument('--fp16', action='store_true', help='FP16 Convert')
	
	args = parser.parse_args()

	if args.mode == "infer":
		infer(args)
	if args.mode == "export":
		export(args)

