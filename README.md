## ___***TrajectoryCrafter: Redirecting View Trajectory for Monocular Videos via Diffusion Models***___
<div align="center">
<img src='assets/title.png' style="height:100px"></img>

 <a href='https://arxiv.org/abs/2409.02048'><img src='https://img.shields.io/badge/arXiv-2409.02048-b31b1b.svg'></a> &nbsp;
 <a href='https://drexubery.github.io/ViewCrafter/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;
 <a href='https://www.youtube.com/watch?v=WGIEmu9eXmU'><img src='https://img.shields.io/badge/Youtube-Video-b31b1b.svg'></a>&nbsp;
 <a href='https://huggingface.co/spaces/Doubiiu/ViewCrafter'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue'></a> &nbsp;


</div>

🤗 If you find TrajectoryCrafter useful, **please help ⭐ this repo**, which is important to Open-Source projects. Thanks!

## 🔆 Introduction

- __[2024-02-23]__: 🔥🔥 Launch the project page and update the arXiv preprint.
- __[2024-02-18]__: Release pretrained models and inference code.

TrajectoryCrafter can generate high-fidelity novel views from <strong>casually captured monocular video</strong>, while also supporting highly precise pose control. Below shows some examples:

<table class="center">
    <tr style="font-weight: bolder;">
        <td>Input Video &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; New Camera Trajectory</td>
    </tr>
  <td>
    <img src=assets/a1.gif style="width: 100%; height: auto;">
  </td>
  </tr>
  <tr>
  <td>
    <img src=assets/a5.gif style="width: 100%; height: auto;">
  </td>
  </tr> 
  <tr>
  <td>
    <img src=assets/a2.gif style="width: 100%; height: auto;">
  </td>
  </tr>
    <tr>
  <td>
    <img src=assets/a4.gif style="width: 100%; height: auto;">
  </td>
  </tr>
</table>


## ⚙️ Setup

### 1. Clone TrajectoryCrafter
```bash
git clone https://github.com/TrajectoryCrafter/TrajectoryCrafter.git
cd TrajectoryCrafter
```
### 2. Setup environments
```bash
conda create -n trajcrafter python=3.10
conda activate trajcrafter
pip install -r requirements.txt
```

### 3. Download pretrained models
Download the following models using git-lfs or [hf_download](https://huggingface.co/docs/huggingface_hub/guides/download):
```bash
# sudo apt install git-lfs
mkdir -p checkpoints/
cd checkpoints
## our pretrained model
git clone https://huggingface.co/TrajectoryCrafter/TrajectoryCrafter
## depth estimation model
git clone https://huggingface.co/tencent/DepthCrafter
git clone https://huggingface.co/stabilityai/stable-video-diffusion-img2vid
## 3D VAE
git clone https://huggingface.co/alibaba-pai/CogVideoX-Fun-V1.1-5b-InP
## caption model
git clone https://huggingface.co/Salesforce/blip2-opt-2.7b
cd ..
```

## 💫 Inference 
### 1. Command line

Run [inference.py](./inference.py) using the following script. Please refer to the [configuration document](docs/config_help.md) and [render document](docs/render_help.md) to set up inference parameters and camera trajectory. 
```bash
  sh run.sh
```

### 2. Local gradio demo

```bash
  python gradio_app.py
```

##  📢 Limitations
Our model excels at handling videos with well-defined objects and clear motion, as demonstrated in the demo videos. However, since it is built upon a pretrained video diffusion model, it may struggle with complex cases that go beyond the generation capabilities of the base model.

## 🤗 Acknowledgements
Many thanks to these open-source projects: [CogVideo-Fun](https://github.com/aigc-apps/CogVideoX-Fun), [ViewCrafter](https://github.com/Drexubery/ViewCrafter) and [DepthCrafter](https://github.com/Tencent/DepthCrafter).

