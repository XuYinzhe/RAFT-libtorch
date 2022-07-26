# RAFT-libtorch
Optical flow method RAFT implemented with Libtorch. <br/>
RAFT: Recurrent All Pairs Field Transforms for Optical Flow<br/>
ECCV 2020<br/>
Zachary Teed and Jia Deng<br/>
[[Code](https://github.com/princeton-vl/RAFT)] [[Paper](https://arxiv.org/pdf/2003.12039.pdf)]

Original libtorch implement codes of python and cpp are from [RATF](https://github.com/chenjianqu/RAFT) and [RAFT_Libtorch](https://github.com/chenjianqu/RAFT_Libtorch). Based on the two repos, I made some modification and implemented them on Windows. Previously, they only have linux version but it is still recommended to read the two repos first.

## Environment
* Windows 10
* GTX 1650Ti 4G
* CUDA 11.4
* cudnn 8
* Visual Studio 2019
* CMake 3.19
* Libtorch 1.12
* torchvision (cpp) 0.13.0 
* Pytorch 1.12
* OpenCV (cpp) 4.5.5
* Eigen3 (cpp)
* spdlog (cpp)

## Quick Start
### 0. Clone the repo
```shell
git clone https://github.com/XuYinzhe/RAFT-libtorch.git
cd RAFT-libtorch
```
### 1. Refer to [[here](https://github.com/princeton-vl/RAFT)] to download pretrained models and save at:
```shell
./python/models
```
### 2. Export Model
```shell
python export_torchscript.py
```
Or setting your path (please modify <input_model> and <output_model>)
```shell
python export_torchscript.py --model=models/<input_model>.pth --export="<output_model>.pt"
```
### 3. CMake cpp code
Modify libraries path in `CMakeLists.txt`
```txt
set(OpenCV_DIR <your path>/opencv455/build)
set(Torch_DIR <your path>/libtorch/share/cmake/Torch)
set(TorchVision_DIR <your path>/torchvision/share/cmake/TorchVision)
```
Then
```shell
cd cpp
mkdir build
cd build
cmake ..
```
### 4. Open `.sln` project in MSVS 
Config include directory
```txt
<your path>\spdlog\include
<your path>\eigen3\include
```
### 5. Modify required path
#### 5.1 Demo frames path 
`config.yaml`
```shell
./cpp/config/config.yaml
```
Only modify DATASET_DIR is enough
```txt
DATASET_DIR: "<your path>\\cpp\\demo\\kitti07"
```
#### 5.2 Config path
`main.cpp`
```shell
./cpp/RAFT/main.cpp
```
Line 55
```cpp
config_file = "<your path>\\cpp\\config\\config.yaml";
```
#### 5.3 Model path
`RAFT_Torch.cpp`
```shell
./cpp/RAFT/src/RAFT_Torch.cpp
```
Line 18
```cpp
raft = std::make_unique<torch::jit::Module>(torch::jit::load("<your path>\\python\\<output_model>.pt"));
```
### 6. Build `.sln` project in release mode
### 7. Copy all `.dll` files to the same directory with `.exe` file
It's my `.dll` list
```txt
2022/07/18  15:34           273,408 asmjit.dll
2022/07/18  15:34           441,856 c10.dll
2022/07/18  15:34           373,248 c10_cuda.dll
2022/07/18  15:34            16,384 caffe2_nvrtc.dll
2022/07/18  15:34       107,136,512 cublas64_11.dll
2022/07/18  15:34       174,900,224 cublasLt64_11.dll
2022/07/18  15:34           497,664 cudart64_110.dll
2022/07/18  15:34           237,568 cudnn64_8.dll
2022/07/18  15:34       129,872,896 cudnn_adv_infer64_8.dll
2022/07/18  15:34        97,293,824 cudnn_adv_train64_8.dll
2022/07/18  15:34       736,718,848 cudnn_cnn_infer64_8.dll
2022/07/18  15:34        81,487,360 cudnn_cnn_train64_8.dll
2022/07/18  15:34        88,405,504 cudnn_ops_infer64_8.dll
2022/07/18  15:34        70,403,584 cudnn_ops_train64_8.dll
2022/07/18  15:34       188,564,992 cufft64_10.dll
2022/07/18  15:34           285,696 cufftw64_10.dll
2022/07/18  15:34         3,848,192 cupti64_2021.1.0.dll
2022/07/18  15:34        60,747,776 curand64_10.dll
2022/07/18  15:34       209,687,552 cusolver64_11.dll
2022/07/18  15:34       215,774,720 cusolverMg64_11.dll
2022/07/18  15:35       224,918,016 cusparse64_11.dll
2022/07/18  15:35         4,655,616 fbgemm.dll
2022/07/18  15:35         1,959,528 libiomp5md.dll
2022/07/18  15:35            41,576 libiompstubs5md.dll
2022/07/18  15:35         5,702,656 nvrtc-builtins64_113.dll
2022/07/18  15:35        32,323,584 nvrtc64_112_0.dll
2022/07/18  15:35            48,128 nvToolsExt64_1.dll
2022/07/18  15:35             9,728 torch.dll
2022/07/18  15:35       233,527,296 torch_cpu.dll
2022/07/18  15:35             9,728 torch_cuda.dll
2022/07/18  15:35         1,337,344 torch_cuda_cpp.dll
2022/07/18  15:35       871,978,496 torch_cuda_cu.dll
2022/07/18  15:35             9,728 torch_global_deps.dll
2022/07/18  15:35           195,072 uv.dll
2022/07/18  15:35            89,088 zlibwapi.dll
```

## Debug helper
If you can read Chinese, following blogs may be useful<br/>
[[blog1](https://blog.csdn.net/zzz_zzz12138/article/details/109138805)]
[[blog2](https://blog.csdn.net/qq_43950348/article/details/115697900?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522161933864616780262518958%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=161933864616780262518958&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_v2~rank_v29-6-115697900.first_rank_v2_pc_rank_v29&utm_term=libtorch+totensor%E6%8A%A5%E9%94%99)]<br/>
In my implement I resize original input images by half, because my cuda is only 4G. If your cuda is large enough, you may modify `Pipeline.cpp`
```shell
./cpp/RAFT/src/Pipeline.cpp
```
Line 34
```cpp
cv::Mat img = resize_image(img_, 0.5);
```
