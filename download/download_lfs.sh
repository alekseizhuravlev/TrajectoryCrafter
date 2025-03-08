sudo apt-get install git-lfs
git lfs install
mkdir -p ./checkpoints/
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

