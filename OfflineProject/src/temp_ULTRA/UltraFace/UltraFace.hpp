//
//  UltraFace.hpp
//  UltraFaceTest
//
//  Created by vealocia on 2019/10/17.
//  Copyright © 2019 vealocia. All rights reserved.
//

#ifndef UltraFace_hpp
#define UltraFace_hpp

//#pragma once
#include <opencv2/core/core.hpp>
#include "gpu.h"
#include "net.h"
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
#include "platform.h"

#define ENABLE_VALIDATION_LAYER 1
#define num_featuremap 4
#define hard_nms 1
#define blending_nms 2 /* mix nms was been proposaled in paper blaze face, aims to minimize the temporal jitter*/

typedef struct FaceInfo {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    cv::Point2f landmark[68];
} FaceInfo;

class UltraFace {
public:
    UltraFace(const std::string &bin_path, const std::string &param_path,
              int input_width, int input_length, int num_thread_ = 4, float score_threshold_ = 0.7, float iou_threshold_ = 0.3, int topk_ = -1);

    ~UltraFace();

    int detect(ncnn::Mat &img, std::vector<FaceInfo> &face_list);
    int detect(ncnn::Mat &img, ncnn::Mat &score_blob32, ncnn::Mat &bbox_blob32,ncnn::Mat &score_blob16,ncnn::Mat &bbox_blob16);
    static int face_embedding(ncnn::Mat &img, std::vector<float> &out);
    void generateBBox(std::vector<FaceInfo> &bbox_collection, ncnn::Mat scores, ncnn::Mat boxes, float score_threshold, int num_anchors);
    ncnn::Mat generate_anchors(int base_size, const ncnn::Mat& ratios, const ncnn::Mat& scales);
    void nms(std::vector<FaceInfo> &input, std::vector<FaceInfo> &output, int type = blending_nms);
    void generate_proposals(const ncnn::Mat& anchors, int feat_stride, const ncnn::Mat& score_blob, const ncnn::Mat& bbox_blob, const ncnn::Mat& landmark_blob, float prob_threshold, std::vector<FaceInfo>& faceobjects);
    void generate_proposals(const ncnn::Mat& anchors, int feat_stride, const ncnn::Mat& score_blob, const ncnn::Mat& bbox_blob, float prob_threshold, std::vector<FaceInfo>& faceobjects);

private:
    static ncnn::Net arcface;
public:

    class init_static // we're defining a nested class named init_static
    {
    public:
        init_static() // the init constructor will initialize our static variable
        {
            arcface.opt.use_vulkan_compute = 1;
//            g_blob_pool_allocator.set_size_compare_ratio(0.0f);
//            g_workspace_pool_allocator.set_size_compare_ratio(0.5f);

//            arcface.opt.blob_allocator = &g_blob_pool_allocator;
//            arcface.opt.workspace_allocator = &g_workspace_pool_allocator;
//        #if NCNN_VULKAN
//            arcface.opt.blob_vkallocator = g_blob_vkallocator;
//            arcface.opt.workspace_vkallocator = g_blob_vkallocator;
//            arcface.opt.staging_vkallocator = g_staging_vkallocator;
//        #endif // NCNN_VULKAN
//            arcface.opt.use_winograd_convolution = true;
//            arcface.opt.use_sgemm_convolution = true;
//            arcface.opt.use_int8_inference = true;
//            arcface.opt.use_fp16_packed = true;
//            arcface.opt.use_fp16_storage = true;
//            arcface.opt.use_fp16_arithmetic = true;
//            arcface.opt.use_int8_storage = true;
//            arcface.opt.use_int8_arithmetic = true;
//            arcface.opt.use_packing_layout = true;
            arcface.load_param("/home/nghiep/Documents/c_face_rec/build/mobilefacenet.param");
            arcface.load_model("/home/nghiep/Documents/c_face_rec/build/mobilefacenet.bin");
        }
    } ;

private:
    static init_static s_initializer;
    ncnn::Net ultraface;

     static ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
     static ncnn::PoolAllocator g_workspace_pool_allocator;
#if NCNN_VULKAN
     static ncnn::VulkanDevice* g_vkdev;
     static ncnn::VkAllocator *g_blob_vkallocator ;
     static ncnn::VkAllocator *g_staging_vkallocator ;
#endif


    int num_thread;
    int image_w;
    int image_h;

    int in_w;
    int in_h;
    int num_anchors;

    int topk;
    float score_threshold;
    float iou_threshold;


    const float mean_vals[3] = {127, 127, 127};
    const float norm_vals[3] = {1.0 / 128, 1.0 / 128, 1.0 / 128};

    const float center_variance = 0.1;
    const float size_variance = 0.2;
    const std::vector<std::vector<float>> min_boxes = {
            {10.0f,  16.0f,  24.0f},
            {32.0f,  48.0f},
            {64.0f,  96.0f},
            {128.0f, 192.0f, 256.0f}};
    const std::vector<float> strides = {8.0, 16.0, 32.0, 64.0};
    std::vector<std::vector<float>> featuremap_size;
    std::vector<std::vector<float>> shrinkage_size;
    std::vector<int> w_h_list;

    std::vector<std::vector<float>> priors = {};
};

#endif /* UltraFace_hpp */
