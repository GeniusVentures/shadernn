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
#pragma once

#include "genericlayer.h"
#include "snn/snn.h"
#include "snn/inferencegraph.h"
#include "modelparser.h"
#include <string>
#include <utility>

namespace snn {
namespace dp { // short for Dynamic Pipeline

struct AddDesc : public CommonLayerDesc {
    std::string activation;
    float leakyReluAlpha;
    void parse(ModelParser& parser, int layerId) {
        CommonLayerDesc::parse(parser, layerId);
        parser.getAddLayer(layerId, activation, leakyReluAlpha);
    }
};

// This is a base class to generates a shader for add function
// It also fuses the activation function
class AddLayer : public ShaderLayer {
public:
    AddLayer(AddDesc&& d): ShaderLayer(d), _desc(std::move(d)) {}
    virtual ~AddLayer() = default;
    InferenceGraph::Transform getOutputScaleDimAdjustment() const override { 
        InferenceGraph::Transform t;
        t.isFixed = 0;
        t.scaleWidth = 1.0f;
        t.scaleHeight = 1.0f;
        t.translateWidth = 0.0f;
        t.translateHeight = 0.0f;
        return t;
        //return {0, {{1.0f, 1.0f, 0.0f, 0.0f}}}; 
    };

protected:
    AddDesc _desc;
};

}; // namespace dp
} // namespace snn
