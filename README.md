# <div align=center> TensorRT_Parser_Python </div>

TensorRT engine **convert** (from ***Onnx*** engine) and **inference** in Python.

The Onnx model can be run on any system with difference platform (Operating system/ CUDA / CuDNN / TensorRT) but take a lot of time to parse.
Convert the Onnx model to TensorRT model (.trt) help you save a lot of parsing time (4-10 min) but can only run on fixed system you've built.

## I. Prerequiste.

- [CUDA/CUDNN/TensorRT Installation Guide](https://github.com/CuteBoiz/Ubuntu_Installation/blob/master/cuda.md)

- **Clone.**
  ```sh
  git clone https://github.com/CuteBoiz/TensorRT_Parser_Python
  cd TensorRT_Parser_Python
  ```
  
## <div align=center> II. Export Onnx engine to TensorRT engine. </div>

```sh
python3 main.py export --weight (--saved_name) (--max_batch_size) (--max_workspace_size) (--fp16) (--input_tensor_name) (--dim) 
```

<details> 
<summary><b>Arguments Details</b></summary>
    
   |Arguments Details   |Type           |Default        |Note
   |---                 |---            |---            |---
   |`--weight`          |`str`          |`required`     |**Path to onnx engine.**
   |`--saved_name`      |`str`          |`'weight_path'.trt`   |**Saved name of trt engine**
   |`--fp16`            |`store_true`   |`false`        |**Use FP16 fast mode (x2 inference time).**
   |`--max_batch_size`  |`int`          |`1`            |**Inference max batchsize.**
   |`--max_workspace_size`| `int`       |`1300`         |**Max workspace size(MB)**
   |`--input_tensor_name`|`str`         |`None`         |**Input tensor name *(dynamic shape input only)*.**
   |`--dim`             |`int_array`    |`None`         |**Input tensor dimension *(dynamic shape input only)*.**

   **Note:** The only GPUs with full-rate FP16 Fast mode performance are Tesla P100, Quadro GP100, and Jetson TX1/TX2.

   **Note:** To get input tensor name/shape of a DL engine: Use [Netron](https://github.com/lutzroeder/netron).
    
</details> 

<details> 
<summary><b>Examples</b></summary>
 
- **Export Onnx engine to TensorRT engine.**
 
  ```sh
  python3 main.py export --weight ../2020_0421_0925.onnx 
  python3 main.py export --weight ../2020_0421_0925.onnx --saved_name model.trt --max_batch_size 10 --fp16
  ```
 
- **Export Onnx engine with Dynamic shape input (batchsize x 3 x 416 x416).**
 
  ```sh
   --input_tensor_name tensorName --dim dims1(,dims2,dims3)  (Does not include batchsize dims)
   python3 main.py export --ds --weight ../2020_0421_0925.onnx --input_tensor_name input_1 --dim 128 128 3
   python3 main.py export --ds --weight ../Keras.onnx --input_tensor_name input:0 --dim 3 640 640 --fp16
   ```
 
</details>

## <div align=center> III. Inference. </div>

```sh
python3 main.py infer --weight --data (--batch_size) (--softmax)
```

<details> 
<summary><b>Arguments Details</b></summary>
    
   |Arguments Details   |Type           |Default        |Note
   |---                 |---            |---            |---
   |`--weight`          |`str`          |`required`     |**Path to trt engine.**
   |`--data`            |`str`          |`required`     |**Path to inference data.**
   |`--batch_size`      |`int`          |`1`            |**Inference batchsize.**
   |`--softmax`         |`store_true`   |`false`        |**Add softmax to output layer.**
    
</details> 

<details> 
<summary><b>Examples</b></summary>
 
```
python3 main.py infer --weight ../2020_0421_0925.trt --data ../Dataset/Train/
python3 main.py infer --weight ../2020_0421_0925.trt --data ../Dataset/Train/ --batch_size 6 --softmax
```
 
</details>


## TO-DO

- [x] Batchsize inference.
- [x] Add missing params (max_workspace_size, gpu).
- [ ] Multiple inputs support.
- [x] Multiple output support.
- [x] Multi-type of inference data (video/folder/image).
