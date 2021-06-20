# TensorRT_Parser_Python

TensorRT engine **convert** (from ***Onnx*** engine) and **inference** in Python.

The Onnx model can be run on any system with difference platform (Operating system/ CUDA / CuDNN / TensorRT) but take a lot of time to parse.
Convert the Onnx model to TensorRT model (.trt) help you save a lot of parsing time (4-10 min) but can only run on fixed system you've built.

## I. Prerequiste.

- [CUDA/CUDNN/TensorRT Installation Guide](https://github.com/CuteBoiz/Ubuntu_Installation/blob/master/cuda.md)
- [Install OpenCV From Source with CUDA support](https://github.com/CuteBoiz/Ubuntu_Installation/blob/master/opencv.md)

## II. Export Onnx model to TensorRT model (.trt).
  - Export:
    ```sh
    python3 main.py export --weight --output --max_batch_size (--fp16)
    ```
    **Note:** The only GPUs with full-rate FP16 Fast mode performance are Tesla P100, Quadro GP100, and Jetson TX1/TX2.

  - Example:
    ```sh
    python3 main.py export --weight ../2020_0421_0925.onnx --output model.trt --max_batch_size 5
    python3 main.py export --weight ../2020_0421_0925.onnx --output model.trt --max_batch_size 10 --fp16
    ```

## III. Export Onnx model to TensorRT model (.trt) with dynamic input shape.
  - Export:
    ```sh
    python3 main.py export --ds --weight --output --max_batch_size --input_tensor_name --dimension (--fp16)
    ```
    **Note:** To get input tensor name and shape of model: Use [Netron](https://github.com/lutzroeder/netron).

  - Example:
    ```sh
    python3 main.py export --ds --weight ../2020_0421_0925.onnx --output model.trt --max_batch_size 5 --input_tensor_name input_1 --dimension 128 128 3
    python3 main.py export --ds --weight ../Keras.onnx --output Keras.trt --max_batch_size 10 --input_tensor_name input --dimension 3 640 640 --fp16
    ```

## IV. Inference:
  - Inference:
    ```sh
    python3 main.py infer --weight --img_path --batch_size
    ```

  - Example:
    ```sh
    python3 main.py infer --weight ../2020_0421_0925.onnx --img_path ../Dataset/Train/ -bs 5
    ```
