/*******************************************************
 *
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of RAFT_Libtorch.
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include <iostream>
#include <memory>
#include <filesystem>
#include <torch/torch.h>
#include <torch/script.h>
#include "src/Config.h"
#include "src/utils.h"
#include "src/Pipeline.h"
#include "src/Visualization.h"
#include "src/RAFT_Torch.h"

namespace fs = std::filesystem;

cv::Mat ReadOneKitti(int index){
    char name[64];
    sprintf(name,"%06d.png",index);
    std::string img0_path=Config::DATASET_DIR+name;
    fmt::print("Read Image:{}\n",img0_path);
    return cv::imread(img0_path);
}


vector<fs::path> ReadImagesNames(const string &path){
    fs::path dir_path(path);
    vector<fs::path> names;
    if(!fs::exists(dir_path))
        return names;
    fs::directory_iterator dir_iter(dir_path);
    for(auto &it : dir_iter){
        cout << it.path() << endl;
        names.push_back(it.path());
    }
    return names;
}


int main(int argc, char **argv) {

    cout << "CUDA: " << torch::cuda::is_available() << endl;
    cout << "Device count: " << torch::cuda::device_count() << endl;
    cout << "Cudnn: " << torch::cuda::cudnn_is_available() << endl;

    string config_file;
    if(argc != 2){
        config_file = "D:\\AA_CourseFile\\2021SummerIntern\\RAFT\\RAFT_Libtorch\\config\\config.yaml";
    }
    else
        config_file = argv[1];

    fmt::print("config_file:{}\n", config_file);
    //RAFT_TorchScript::Ptr raft_torchscript;
    //RAFT_Torch::Ptr raft_torch;
    RAFT_TorchScript::Ptr raft_torch;
    try{
        cout << "Initialize Config" << endl;
        Config cfg(config_file);
        cout << "Create RAFT_Torch" << endl;
        raft_torch = std::make_unique<RAFT_TorchScript>();
        //raft_torch = std::make_unique<RAFT_Torch>();
    }
    catch(std::runtime_error &e){
        sgLogger->critical(e.what());
        cerr<<e.what()<<endl;
        return -1;
    }

    cout << "Create TicToc" << endl;
    TicToc tt;

    cout << "Read all image fills" << endl;
    auto names = ReadImagesNames(Config::DATASET_DIR);
    std::sort(names.begin(),names.end());

    cout << "Read img0" << endl;
    cv::Mat img0,img1;
    img0 = cv::imread(names[0].string());
    //img0 = cv::imread(names[0].string(), cv::IMREAD_COLOR);

    cout << "Process img0" << endl;
    Tensor flow;
    Tensor tensor0 = Pipeline::process(img0);//(1,3,376, 1232),值大小从-1到1
    cout << tensor0.device() << endl;
    cout << tensor0.sizes() << endl;

    Tensor final_pred, find_image;

    cout << "iter numbers: " << names.size() << endl;
    for(int index=1; index <names.size();++index)
    {
        cout << "Read img" << index << endl;
        img1 = cv::imread(names[index].string());
        //img1 = cv::imread(names[index].string(), cv::IMREAD_COLOR);
        fmt::print(names[index].string()+"\n");
        if(img1.empty()){
            cerr<<"Read image:"<<index<<" failure"<<endl;
            break;
        }
        cout << "Process img" << index << endl;
        tt.Tic();
        Tensor tensor1 = Pipeline::process(img1);//(1,3,376, 1232)

        Debugs("process:{} ms", tt.TocThenTic());

        cout << "Start to predict" << endl;
        Tensor prediction = raft_torch->Forward_flow2(tensor0, tensor1);
        //vector<Tensor> prediction = raft_torch->Forward(tensor0, tensor1);

        Debugs("prediction:{} ms", tt.TocThenTic());

        if (index == names.size() - 1) {
            final_pred = prediction.squeeze();
            find_image = (tensor1.squeeze() + 1.) / 2.;
        }
/*
        torch::Tensor tensor1_raw = (tensor1.squeeze()+1.)/2.;
        flow = prediction.back();//[1,2,h,w]
        flow = flow.squeeze();

        cv::Mat flow_show = visual_flow_image(tensor1_raw,flow);
        //cv::Mat flow_show = visual_flow_image(flow);

        cv::imshow("flow",flow_show);
        if(auto order=(cv::waitKey(100) & 0xFF); order == 'q')
            break;
        else if(order==' ')
            cv::waitKey(0);*/

        tensor0 = tensor1;
    }

    //visualization
    cv::Mat flow_show = visual_flow_image(final_pred);
    cv::imshow("flow", flow_show);
    cv::waitKey(0);


    return 0;
}
