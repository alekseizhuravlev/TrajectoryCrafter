## ___***TrajectoryCrafter: Generative View Trajectory Redirection for Monocular Videos***___
<div align="center">

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
        <td>Input Video &emsp;&emsp;&emsp;&emsp;&emsp; Generated Novel Views</td>
    </tr>
  <td>
    <img src=assets/a1.gif style="width: 100%; height: auto;">
  </td>
  </tr>
  <tr>
  <td>
    <img src=assets/a3.gif style="width: 100%; height: auto;">
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




## 🧰 Models

|Model|Resolution|Frames|GPU Mem. & Inference Time (tested on a 40G A100, ddim 50 steps)|Checkpoint|Description|
|:---------|:---------|:--------|:--------|:--------|:--------|
|ViewCrafter_25|576x1024|25| 23.5GB & 120s (`perframe_ae=True`)|[Hugging Face](https://huggingface.co/Drexubery/ViewCrafter_25/blob/main/model.ckpt)|Used for single view NVS, can also adapt to sparse view NVS|
|ViewCrafter_25_sparse|576x1024|25| 23.5GB & 120s (`perframe_ae=True`)|[Hugging Face](https://huggingface.co/Drexubery/ViewCrafter_25_sparse/blob/main/model_sparse.ckpt)|Used for sparse view NVS|
|ViewCrafter_16|576x1024|16| 18.3GB & 75s (`perframe_ae=True`)|[Hugging Face](https://huggingface.co/Drexubery/ViewCrafter_16/blob/main/model.ckpt)|16 frames model, used for ablation|
|ViewCrafter_25_512|320x512|25| 13.8GB & 50s (`perframe_ae=True`)|[Hugging Face](https://huggingface.co/Drexubery/ViewCrafter_25_512/blob/main/model.ckpt)|512 resolution model, used for ablation|

## ⚙️ Setup

### 1. Clone ViewCrafter
```bash
git clone https://github.com/Drexubery/ViewCrafter.git
cd ViewCrafter
```
### 2. Installation

```bash
# Create conda environment
conda create -n viewcrafter python=3.9.16
conda activate viewcrafter
pip install -r requirements.txt

# Install PyTorch3D
conda install https://anaconda.org/pytorch3d/pytorch3d/0.7.5/download/linux-64/pytorch3d-0.7.5-py39_cu117_pyt1131.tar.bz2

# Download pretrained DUSt3R model
mkdir -p checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth -P checkpoints/

```
> [!NOTE]
> If you use a high PyTorch version (like torch 2.4), it may cause CUDA OOM error. Please refer to [these issues](https://github.com/Drexubery/ViewCrafter/issues/23#issuecomment-2396131121) for solutions.

## 💫 Inference 
### 1. Command line
### Single view novel view synthesis
(1) Download pretrained [ViewCrafter_25](https://huggingface.co/Drexubery/ViewCrafter_25/blob/main/model.ckpt) and put the `model.ckpt` in `checkpoints/model.ckpt`. \
(2) Run [inference.py](./inference.py) using the following script. Please refer to the [configuration document](docs/config_help.md) and [render document](docs/render_help.md) to set up inference parameters and camera trajectory. 
```bash
  sh run.sh
```
### Sparse view novel view synthesis
(1) Download pretrained [ViewCrafter_25_sparse](https://huggingface.co/Drexubery/ViewCrafter_25_sparse/blob/main/model_sparse.ckpt) and put the `model_sparse.ckpt` in `checkpoints/model_sparse.ckpt`. ([ViewCrafter_25_sparse](https://huggingface.co/Drexubery/ViewCrafter_25_sparse/blob/main/model_sparse.ckpt) is specifically trained for the sparse view NVS task and performs better than [ViewCrafter_25](https://huggingface.co/Drexubery/ViewCrafter_25/blob/main/model.ckpt) on this task) \
(2) Run [inference.py](./inference.py) using the following script. Adjust the `--bg_trd` parameter to clean the point cloud; higher values will produce a cleaner point cloud but may create holes in the background.
```bash
  sh run_sparse.sh
```
### 2. Local Gradio demo

Download pretrained [ViewCrafter_25](https://huggingface.co/Drexubery/ViewCrafter_25/blob/main/model.ckpt) and put the `model.ckpt` in `checkpoints/model.ckpt`, then run:
```bash
  python gradio_app.py 
```

## 📈 Evaluation

We provide a demo script to evaluate single-view novel view synthesis:
```bash
  sh run_eval.sh
```
The input should be a folder containing frames from your test video. We use the first frame as the reference image and the subsequent frames as target novel views.

## 😉 Citation
Please consider citing our paper if our code is useful:
```bib
  @article{yu2024viewcrafter,
    title={ViewCrafter: Taming Video Diffusion Models for High-fidelity Novel View Synthesis},
    author={Yu, Wangbo and Xing, Jinbo and Yuan, Li and Hu, Wenbo and Li, Xiaoyu and Huang, Zhipeng and Gao, Xiangjun and Wong, Tien-Tsin and Shan, Ying and Tian, Yonghong},
    journal={arXiv preprint arXiv:2409.02048},
    year={2024}
  }
```

<a name="disc"></a>
## 📢 Disclaimer
⚠️This is an open-source research exploration rather than a commercial product, so it may not meet all your expectations. Due to the variability of the video diffusion model, you may encounter failure cases. Try using different seeds and adjusting the render configs if the results are not desirable.
Users are free to create videos using this tool, but they must comply with local laws and use it responsibly. The developers do not assume any responsibility for potential misuse.
****

