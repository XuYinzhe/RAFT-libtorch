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
### 4. Open .sln project in MSVS 
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
