# TensorRT_Parser_Python

TensorRT engine **convert** (from ***Onnx*** engine) and **inference** in Python.

The Onnx model can be run on any system with difference platform (Operating system/ CUDA / CuDNN / TensorRT) but take a lot of time to parse.
Convert the Onnx model to TensorRT model (.trt) help you save a lot of parsing time (4-10 min) but can only run on fixed system you've built.

## I. Prerequiste.

- [CUDA/CUDNN/TensorRT Installation Guide](https://github.com/CuteBoiz/Ubuntu_Installation/blob/master/cuda.md)
- [Install OpenCV From Source with CUDA support](https://github.com/CuteBoiz/Ubuntu_Installation/blob/master/opencv.md)

## II. Export Onnx model to TensorRT model (.trt).
  - **Export:**
    ```sh
    python3 main.py export --weight --name (--max_batch_size) (--fp16)
    ```
    - ***Arguments***
      - `weight`: Path to Onnx model.
      - `name`: Saved name of output TensorRT engine.
      - `max_batch_size`: Max inference batch size *(default = 1)*.
      - `fp16`: Use fp16 fast mode *(default = False)*.
      
      **Note:** The only GPUs with full-rate FP16 Fast mode performance are Tesla P100, Quadro GP100, and Jetson TX1/TX2.

  - **Example:**
    ```sh
    python3 main.py export --weight ../2020_0421_0925.onnx --name model.trt --max_batch_size 5
    python3 main.py export --weight ../2020_0421_0925.onnx --name model.trt --max_batch_size 10 --fp16
    ```

## III. Export Onnx model to TensorRT model (.trt) with dynamic input shape.
  - **Export:**
    ```sh
    python3 main.py export --ds --weight --name (--max_batch_size) --input_tensor_name --dimension (--fp16)
    ```
    - ***Arguments***
      - `ds`: Enable dynamic input shape convert.
      - `weight`: Path to Onnx model.
      - `name`: Saved name of output TensorRT engine.
      - `input_tensor_name`: Name of Onnx's first layer.
      - `dimension`: Dimension of Onnx's first layer.
      - `max_batch_size`: Max inference batch size *(default = 1)*.
      - `fp16`: Use fp16 fast mode *(default = False)*.
      
      **Note:** The only GPUs with full-rate FP16 Fast mode performance are Tesla P100, Quadro GP100, and Jetson TX1/TX2.  
      
      **Note:** To get input tensor name and shape of model: Use [Netron](https://github.com/lutzroeder/netron).

  - **Example:**
    ```sh
    python3 main.py export --ds --weight ../2020_0421_0925.onnx --name model.trt --max_batch_size 5 --input_tensor_name input_1 --dimension 128 128 3
    python3 main.py export --ds --weight ../Keras.onnx --name Keras.trt --max_batch_size 10 --input_tensor_name input --dimension 3 640 640 --fp16
    ```

## IV. Inference:
  - **Inference:**
    ```sh
    python3 main.py infer --weight --path (--batch_size) (--softmax)
    ```
    - ***Arguments***
      - `weight`: Path to TensorRT model.
      - `path`: Path to inference images folder.
      - `batch_size`: Inference batch size *(default = 1)*.
      - `softmax`: Use softmax.
      
  - **Example:**
    ```sh
    python3 main.py infer --weight ../2020_0421_0925.onnx --path ../Dataset/Train/
    python3 main.py infer --weight ../2020_0421_0925.onnx --path ../Dataset/Train/ -bs 6 --softmax
    ```

## V.TO-DO

- [ ] Batchsize inference