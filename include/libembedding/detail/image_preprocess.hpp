/*
 * libembedding - detail/image_preprocess.hpp
 * Image preprocessing pipeline (resize, crop, normalize, HWC->CHW)
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_DETAIL_IMAGE_PREPROCESS_HPP
#define LIBEMBEDDING_DETAIL_IMAGE_PREPROCESS_HPP

#ifndef LIBEMBEDDING_NO_IMAGE

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

/* stb_image and stb_image_resize are expected to be available */
#include "stb/stb_image.h"
#include "stb/stb_image_resize2.h"

namespace lembed { namespace detail {

/* Image data in CHW float format */
struct ImageTensor {
    std::vector<float> data; /* CHW layout */
    int channels;
    int height;
    int width;
};

/* Load image from file, resize to target_size, center crop, normalize, convert to CHW */
inline ImageTensor load_and_preprocess_image(
        const std::string& path,
        int target_size = 224,
        const float mean[3] = nullptr,
        const float std_dev[3] = nullptr) {

    /* Default ImageNet normalization */
    static const float default_mean[3] = { 0.485f, 0.456f, 0.406f };
    static const float default_std[3]  = { 0.229f, 0.224f, 0.225f };
    if (!mean) mean = default_mean;
    if (!std_dev) std_dev = default_std;

    /* Load image */
    int w, h, c;
    unsigned char* img = stbi_load(path.c_str(), &w, &h, &c, 3);
    if (!img) {
        throw std::runtime_error("Failed to load image: " + path);
    }
    c = 3; /* force RGB */

    /* Resize shorter side to target_size, maintaining aspect ratio */
    int new_w, new_h;
    if (w < h) {
        new_w = target_size;
        new_h = (int)((float)h / w * target_size);
    } else {
        new_h = target_size;
        new_w = (int)((float)w / h * target_size);
    }

    std::vector<unsigned char> resized(new_w * new_h * 3);
    stbir_resize_uint8_linear(
        img, w, h, 0,
        resized.data(), new_w, new_h, 0,
        (stbir_pixel_layout)3 /* STBIR_RGB */);
    stbi_image_free(img);

    /* Center crop to target_size x target_size */
    int crop_x = (new_w - target_size) / 2;
    int crop_y = (new_h - target_size) / 2;

    ImageTensor result;
    result.channels = 3;
    result.height = target_size;
    result.width = target_size;
    result.data.resize(3 * target_size * target_size);

    /* Convert to CHW float, rescale [0,255] -> [0,1], normalize */
    for (int ch = 0; ch < 3; ch++) {
        for (int y = 0; y < target_size; y++) {
            for (int x = 0; x < target_size; x++) {
                int src_x = crop_x + x;
                int src_y = crop_y + y;
                float pixel = (float)resized[(src_y * new_w + src_x) * 3 + ch] / 255.0f;
                pixel = (pixel - mean[ch]) / std_dev[ch];
                result.data[ch * target_size * target_size + y * target_size + x] = pixel;
            }
        }
    }

    return result;
}

/* Load image from memory buffer */
inline ImageTensor load_and_preprocess_image_bytes(
        const unsigned char* data, int data_size,
        int target_size = 224,
        const float mean[3] = nullptr,
        const float std_dev[3] = nullptr) {

    static const float default_mean[3] = { 0.485f, 0.456f, 0.406f };
    static const float default_std[3]  = { 0.229f, 0.224f, 0.225f };
    if (!mean) mean = default_mean;
    if (!std_dev) std_dev = default_std;

    int w, h, c;
    unsigned char* img = stbi_load_from_memory(data, data_size, &w, &h, &c, 3);
    if (!img) {
        throw std::runtime_error("Failed to decode image from memory");
    }
    c = 3;

    int new_w, new_h;
    if (w < h) {
        new_w = target_size;
        new_h = (int)((float)h / w * target_size);
    } else {
        new_h = target_size;
        new_w = (int)((float)w / h * target_size);
    }

    std::vector<unsigned char> resized(new_w * new_h * 3);
    stbir_resize_uint8_linear(
        img, w, h, 0,
        resized.data(), new_w, new_h, 0,
        (stbir_pixel_layout)3);
    stbi_image_free(img);

    int crop_x = (new_w - target_size) / 2;
    int crop_y = (new_h - target_size) / 2;

    ImageTensor result;
    result.channels = 3;
    result.height = target_size;
    result.width = target_size;
    result.data.resize(3 * target_size * target_size);

    for (int ch = 0; ch < 3; ch++) {
        for (int y = 0; y < target_size; y++) {
            for (int x = 0; x < target_size; x++) {
                int src_x = crop_x + x;
                int src_y = crop_y + y;
                float pixel = (float)resized[(src_y * new_w + src_x) * 3 + ch] / 255.0f;
                pixel = (pixel - mean[ch]) / std_dev[ch];
                result.data[ch * target_size * target_size + y * target_size + x] = pixel;
            }
        }
    }

    return result;
}

}} /* namespace lembed::detail */

#endif /* LIBEMBEDDING_NO_IMAGE */
#endif /* LIBEMBEDDING_DETAIL_IMAGE_PREPROCESS_HPP */
