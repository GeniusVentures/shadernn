/* Copyright (C) 2020 - 2022 OPPO. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "pch.h"
#include "separableconvolution.h"
#include "layerFactory.h"
#include "inferencepass.h"
#include <string>
#include <vector>
#include <algorithm>
#include <utility>

using namespace snn;
using namespace snn::dp;

void SeparableConv2DLayer::getPaddingOffset(uint32_t (&offsets)[4]) const {
    std::string paddingT = _desc.paddingT;
    std::string paddingB = _desc.paddingB;
    std::string paddingL = _desc.paddingL;
    std::string paddingR = _desc.paddingR;
    bool isdigit         = std::all_of(paddingT.begin(), paddingT.end(), ::isdigit);
    if (isdigit) {
        offsets[0] = std::stoul(paddingT);
        offsets[1] = std::stoul(paddingB);
        offsets[2] = std::stoul(paddingL);
        offsets[3] = std::stoul(paddingR);
    } else {
        if (paddingT == "valid" || paddingT == "none") {
            offsets[0] = 0;
            offsets[1] = 0;
            offsets[2] = 0;
            offsets[3] = 0;
        } else {
            if (_desc.kernelSize > 1) {
                offsets[0] = std::max(static_cast<uint32_t>(_desc.kernelSize / 2), (uint32_t) 1);
                offsets[1] = std::max(static_cast<uint32_t>(_desc.kernelSize / 2), (uint32_t) 1);
                offsets[2] = std::max(static_cast<uint32_t>(_desc.kernelSize / 2), (uint32_t) 1);
                offsets[3] = std::max(static_cast<uint32_t>(_desc.kernelSize / 2), (uint32_t) 1);
                if (_desc.kernelSize % 2 == 0) {
                    offsets[0] = offsets[0] - 1;
                    offsets[2] = offsets[2] - 1;
                }
            } else {
                offsets[0] = 0;
                offsets[1] = 0;
                offsets[2] = 0;
                offsets[3] = 0;
            }
        }
    }
}

InferenceGraph::Transform SeparableConv2DLayer::getOutputScaleDimAdjustment() const {
    uint32_t offset[4];
    getPaddingOffset(offset);
    float scale       = 1.0f / static_cast<float>(_desc.stride);
    float translation = 0.0f;
    if (_desc.kernelSize % 2 != 0) {
        translation = 1.0f + (static_cast<float>(offset[0] + offset[1]) - static_cast<float>(_desc.kernelSize)) / static_cast<float>(_desc.stride);
    } else {
        translation = 1.0f + (static_cast<float>(offset[0] + offset[1] - 1) - static_cast<float>(_desc.kernelSize)) / static_cast<float>(_desc.stride);
    }
    InferenceGraph::Transform t;
    t.isFixed = 0;
    t.scaleWidth = scale;
    t.scaleHeight = scale;
    t.translateWidth = translation;
    t.translateHeight = translation;
    return t;
    //return {0, {{scale, scale, translation, translation}} };
}

void SeparableConv2DLayer::getOutputDims(uint32_t& width, uint32_t& height, uint32_t& depth) const {
    uint32_t paddingOffsets[4];
    getPaddingOffset(paddingOffsets);
    for (auto& dim : inputDims) {
        width  = (dim.width - _desc.kernelSize + paddingOffsets[0] + paddingOffsets[2]) / _desc.stride + 1;
        height = (dim.height - _desc.kernelSize + paddingOffsets[1] + paddingOffsets[3]) / _desc.stride + 1;
        depth  = dim.depth;
        break;
    }
}

bool SeparableConv2DLayer::oihw2hwo4i4(std::vector<cv::Mat> inputWeights, std::vector<float>& outVec, int inChannels,
    int outChannels, int fw, int fh, int unit) {
    (void) inChannels;
    int alignedWeightSize = ROUND_UP(outChannels, unit) * fw * fh;

    outVec.clear();
    outVec.resize(alignedWeightSize);
    std::fill(outVec.begin(), outVec.end(), 0);

    float* out    = (float*) outVec.data();
    int planeSize = ROUND_UP(outChannels, unit) * fw;
    for (int b = 0; b < outChannels; ++b) {
        int b_4 = b / unit;
        int mx  = b % unit;
        for (int y = 0; y < fh; ++y) {
            for (int x = 0; x < fw; ++x) {
                int base                                 = y * planeSize;
                int inSize                               = ROUND_UP(outChannels, unit); // in the number of floats
                out[base + inSize * x + b_4 * unit + mx] = inputWeights[b].at<float>(y * fw + x);
            }
        }
    }
    return 0;
}
