import sys
import os
import copy
from datetime import datetime
import torch
import numpy as np

sys.path.append('/home/azhuravl/work/TrajectoryCrafter/notebooks/06_10_25_vggt')
sys.path.append('/home/azhuravl/work/TrajectoryCrafter/notebooks/12_11_25_consistent_depth/autoregressive_alignment')

from utils_autoregressive import load_video_frames, generate_traj_specified, TrajCrafterAutoregressive
from parsing import get_parser
from autoregressive_loop_alignment import autoregressive_loop, estimate_depth_without_alignment
from autoregressive_loop_alignment import video_to_pcs, invert_depth_with_scale, imagenet_to_0_1
from warper_point_cloud import GlobalPointCloudWarper


sys.path.append('/home/azhuravl/work/Video-Depth-Anything')
sys.path.append('/home/azhuravl/work/Video-Depth-Anything/video_depth_anything/util')

# Video Depth Anything imports
from video_depth_anything.video_depth import VideoDepthAnything
from utils.dc_utils import read_video_frames

sys.path.append('/home/azhuravl/work/TrajectoryCrafter/notebooks/12_11_25_consistent_depth/depth_alignment')

# Depth trainer import
from depth_trainer import DepthAlignmentTrainer
from consistent_depth import prepare_frames, denormalize_rgb
    

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def setup_opts():
    sys.argv = [
        "",
        "--video_path", "/home/azhuravl/nobackup/DAVIS_testing/trainval/rhino.mp4",
        "--n_splits", "4",
        "--overlap_frames", "0",
        "--radius", "0",
        "--mode", "gradual",
        "--video_length", "32",
        # "--sample_size", "266", "462"
    ]

    parser = get_parser()
    opts_base = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    video_basename = os.path.splitext(os.path.basename(opts_base.video_path))[0]

    # Setup
    opts_base.weight_dtype = torch.bfloat16
    opts_base.exp_name = f"{video_basename}_{timestamp}_autoregressive"
    opts_base.save_dir = os.path.join(opts_base.out_dir, opts_base.exp_name)

    # Create TrajCrafterVisualization instance for autoregressive generation
    radius = opts_base.radius

    pose = [90, 0, 0, 0, 1]
    # name = '120_0_0_0_3', make it infer values from pose
    name = f"{pose[0]}_{pose[1]}_{pose[2]}_{pose[3]}_{pose[4]}"


    print(f"\n=== Running Autoregressive {name} ===")
    opts = copy.deepcopy(opts_base)
    opts.exp_name = f"{video_basename}_{timestamp}_{name}_auto_s{opts_base.n_splits}"
    opts.save_dir = os.path.join(opts.out_dir, opts.exp_name)
    opts.camera = "target"
    opts.target_pose = pose
    opts.traj_txt = 'test/trajs/loop2.txt'

    # Make directories
    os.makedirs(opts.save_dir, exist_ok=True)

    return opts


def setup_vda():
    
    class ArgsVDA:
        def __init__(self):
            self.input_video = '/home/azhuravl/scratch/datasets_latents/monkaa_1000/000/videos/input_video.mp4'
            self.output_dir = '/home/azhuravl/work/Video-Depth-Anything/outputs'
            self.input_size = 256
            self.max_res = 1280
            self.encoder = 'vitl'
            self.max_len = -1
            self.target_fps = -1
            self.metric = False
            self.fp32 = False
            self.grayscale = False
            self.save_npz = False
            self.save_exr = False
            self.focal_length_x = 470.4
            self.focal_length_y = 470.4

    args_vda = ArgsVDA()
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }
    checkpoint_name = 'video_depth_anything'

    video_depth_anything = VideoDepthAnything(**model_configs[args_vda.encoder], metric=args_vda.metric)
    video_depth_anything.load_state_dict(torch.load(
        f'/home/azhuravl/work/Video-Depth-Anything/checkpoints/{checkpoint_name}_{args_vda.encoder}.pth', 
        map_location='cpu', weights_only=True), strict=True)
    video_depth_anything = video_depth_anything.to(DEVICE).eval()

    # disable grad for video_depth_anything
    for param in video_depth_anything.parameters():
        param.requires_grad = False
        
    return video_depth_anything, args_vda




if __name__ == '__main__':
    
    #########################################
    # Initialize models
    #########################################
    
    opts = setup_opts()
    
    vis_crafter = TrajCrafterAutoregressive(opts)
    funwarp = GlobalPointCloudWarper(device=opts.device, max_points=2000000)
    
    # TODO: depth estimator + alignment
    video_depth_anything, args_vda = setup_vda()
    

    depth_trainer = DepthAlignmentTrainer(
        video_depth_anything,
        lr=2e-3,
        device=opts.device
    )
    
    
    ###########################################
    # Read video, normalize to imagenet
    ###########################################
    

    frames, target_fps = read_video_frames(
        opts.video_path,
        32, args_vda.target_fps, args_vda.max_res
        ) # (32, 480, 854, 3) uint8 0 255
    # reverse the frames
    # frames = frames[::-1]


    frames_resized_im, orig_dims = prepare_frames(
        frames, input_size=opts.sample_size, normalize_imagenet=True,
        )  # torch.Size([32, 3, 266, 462]) torch.float32 tensor(-2.2437) tensor(2.6739)
    frames_resized_im = frames_resized_im.squeeze(0)
    
    
    print('frames_resized_im', frames_resized_im.shape, frames_resized_im.dtype, frames_resized_im.min(), frames_resized_im.max())

    # frames_resized_denorm, orig_dims = prepare_frames(
    #     frames, input_size=args_vda.input_size, normalize_imagenet=False
    # )
    # frames_resized_denorm = torch.clamp(frames_resized_denorm, 0.0, 1.0).squeeze(0)
    # torch.Size([1, 32, 3, 266, 462]) torch.float32 tensor(0.) tensor(1.)

    # Estimate inverse depths with VDA
    # with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    #     depths_input_inv = video_depth_anything.forward(
    #         frames_resized_im.to(opts.device, dtype=torch.bfloat16).unsqueeze(0)
    #         ).permute(1, 0, 2, 3)  # [1, T, H, W] -> [T, 1, H, W]


    # depths_input_inv = estimate_depth_with_padding(
    #     # frames_resized_im, 
    #     frames_resized_denorm * 255.0,
    #     video_depth_anything, 
    #     opts.device,
    #     multiple_of=14
    # )
    
    
    ###############################################
    # Estimate depth - will be used as anchor
    ###############################################
    
    depth_scale = 10000.0  # set depth scale
    
    depths_input = estimate_depth_without_alignment(
        frames_resized_im,
        depth_trainer,
        depth_scale,
    )
    
    depths_input_inv = invert_depth_with_scale(depths_input, depth_scale)

    print('depths_input_inv', depths_input_inv.shape, depths_input_inv.dtype, depths_input_inv.min(), depths_input_inv.max())
    print('depths_input', depths_input.shape, depths_input.dtype, depths_input.min(), depths_input.max())
    
    # save several depth frames using matplotlib, with colorbar
    import matplotlib.pyplot as plt
    for i in range(0, depths_input_inv.shape[0], 5):
        plt.imshow(depths_input_inv[i, 0].cpu().numpy(), cmap='plasma')
        plt.axis('off')
        plt.colorbar(shrink=0.4)
        plt.savefig(f'{opts.save_dir}/inv_depth_frame_{i}.png')
        plt.clf()


    ##########################################
    # Camera trajectories
    ##########################################

    radius = (
        depths_input[0, 0, depths_input.shape[-2] // 2, depths_input.shape[-1] // 2].cpu()
        * opts.radius_scale
    )
    # radius = min(radius, 5)
    
    print(f"Estimated radius: {radius}")

    c2ws_anchor = torch.tensor([ 
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
        ]).unsqueeze(0).to(opts.device)

    c2ws_target = generate_traj_specified(
        c2ws_anchor, 
        opts.target_pose, 
        opts.video_length * opts.n_splits, 
        opts.device
    )
    c2ws_target[:, 2, 3] += radius

    c2ws_init = c2ws_target[0].repeat(opts.video_length, 1, 1)
    traj_segments = c2ws_target.view(opts.n_splits, opts.video_length, 4, 4)
    
    
    ###############################################
    # Initialize global point clouds
    ###############################################
    
    frames_resized_tensor = imagenet_to_0_1(frames_resized_im).to(opts.device) * 2.0 - 1.0
    
    global_pcs, global_colors = video_to_pcs(
        frames_resized_tensor,
        depths_input,
        intrinsics_torch=vis_crafter.K,
        extrinsics_torch=c2ws_init,
        funwarp=funwarp,
    )
    
    ###############################################
    # Autoregressive loop
    ###############################################
    
    frames_source_im = frames_resized_im

    poses_source = c2ws_init
    poses_target = traj_segments[0]
        
    for i in range(opts.n_splits):
        
        segment_dir_autoreg, global_pcs, global_colors = autoregressive_loop(
            frames_source_im,
            poses_source,
            poses_target,
            global_pcs,
            global_colors,
            radius,
            opts,  
            vis_crafter,
            funwarp,
            video_depth_anything,
            depth_trainer,
            args_vda,
            depth_scale,
            i
        )
        
        # next pose in sequence
        poses_source = poses_target
        poses_target = traj_segments[i+1] if i + 1 < opts.n_splits else None
        
        
        ###############################################
        # Read generated video frames for next iteration
        ###############################################
        
        # TODO: remove reading video
        
        print('frames_source_im', frames_source_im.shape, frames_source_im.dtype, frames_source_im.min(), frames_source_im.max())
        # print('frames_target_im', frames_target_im.shape, frames_target_im.dtype, frames_target_im.min(), frames_target_im.max())

        frames, target_fps = read_video_frames(
            segment_dir_autoreg + '/gen.mp4',
            32, args_vda.target_fps, args_vda.max_res
            ) # (32, 480, 854, 3) uint8 0 255


        print('frames', frames.shape, frames.dtype, frames.min(), frames.max())

        frames_resized_im, orig_dims = prepare_frames(
            frames, input_size=opts.sample_size, normalize_imagenet=True, 
            )  # torch.Size([32, 3, 266, 462]) torch.float32 tensor(-2.2437) tensor(2.6739)
        frames_resized_im = frames_resized_im.squeeze(0)
        
        print('frames_resized_im', frames_resized_im.shape, frames_resized_im.dtype, frames_resized_im.min(), frames_resized_im.max())

        frames_source_im = frames_resized_im
        
        
        
        
        
        

        # break
            
            
            
# segment_dir_autoreg = autoregressive_loop(
#     frames_source_im,
#     frames_target_im,
#     poses_source,
#     poses_target,
#     radius,
#     opts,  
#     vis_crafter,
#     funwarp,
#     video_depth_anything,
#     depth_trainer,
#     args_vda,
#     depth_scale,
# )

# # concatenate frames_source_im and frames_target_im, handling None initially
# if frames_source_im is None:
#     frames_source_im = frames_target_im
# else:
#     frames_source_im = torch.cat([frames_source_im, frames_target_im], axis=0)
