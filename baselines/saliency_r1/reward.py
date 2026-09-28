# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import numpy as np
def think_saliency_reward(valid_list, saliency_map, bbox_list, **kwargs) -> list[float]:
    overlap_list = []
    for saliency, bbox in zip(saliency_map, bbox_list):
        bbox = [float(i) for i in bbox[1:-1].split(",")]
        point = int(saliency.shape[1] * bbox[0]), int(saliency.shape[0] * bbox[1]), int(
            saliency.shape[1] * bbox[2]), int(saliency.shape[1] * bbox[3])
        overlap_list.append(np.sum(saliency[point[1]:point[3]+1, point[0]:point[2]+1]) / np.sum(saliency))
    return [i * j for i, j in zip(overlap_list, valid_list)]
