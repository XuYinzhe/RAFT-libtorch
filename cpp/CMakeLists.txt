cmake_minimum_required(VERSION 3.19)
project(RAFT_Libtorch)

set(CMAKE_CXX_STANDARD 17)

set(OpenCV_DIR D:/OpenCV/opencv455/build)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(Torch_DIR D:/libtorch/libtorch-win-shared-with-deps-1.12.0+cu113/libtorch/share/cmake/Torch)
set(TorchVision_DIR D:/torchvision/torchvision/share/cmake/TorchVision)

find_package(Torch REQUIRED)
find_package(TorchVision REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})

aux_source_directory(RAFT/src src_SOURCES)

add_executable(RAFT_Libtorch RAFT/main.cpp ${src_SOURCES})
target_link_libraries(RAFT_Libtorch ${OpenCV_LIBRARIES}  ${TORCH_LIBRARIES}
        TorchVision::TorchVision )

