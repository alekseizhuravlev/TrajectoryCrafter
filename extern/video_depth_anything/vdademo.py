# Copyright (2025) Bytedance Ltd. and/or its affiliates 

# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at 

#     http://www.apache.org/licenses/LICENSE-2.0 

# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License. 
import argparse
import numpy as np
import os
import torch
from extern.video_depth_anything.video_depth import VideoDepthAnything

class VDADemo:
    def __init__(
        self,
        pre_train_path: str,
        encoder: str = "vitl",
        device: str = "cuda:0",
    ):

        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }

        self.video_depth_anything = VideoDepthAnything(**model_configs[encoder])
        self.video_depth_anything.load_state_dict(torch.load(pre_train_path, map_location='cpu'), strict=True)
        self.video_depth_anything = self.video_depth_anything.to(device).eval()
        self.device = device

    def infer(
        self,
        frames,
        near,
        far,
        input_size = 518,
        target_fps = -1,
    ):
        if frames.max() < 2.:
            frames = frames*255. 
            
        with torch.inference_mode():
            depths, fps = self.video_depth_anything.infer_video_depth(frames, target_fps, input_size, self.device)

        depths = torch.from_numpy(depths).unsqueeze(1) # 49 576 1024 ->
        depths[depths < 1e-5] = 1e-5  
        depths = 10000. / depths
        depths = depths.clip(near, far)


        return depths


    


