import torch
import accelerate
from accelerate.state import AcceleratorState
from transformers.utils import ContextManagers
from videox_fun.models import (AutoencoderKLCogVideoX,
                              CogVideoXTransformer3DModel, T5EncoderModel,
                              T5Tokenizer)
from videox_fun.utils.lora_utils import create_network
from models.crosstransformer3d import CrossTransformer3DModel
from tqdm import tqdm

def setup_models(args, weight_dtype):
    """Setup and load all models"""
    
    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    # Load tokenizer
    tokenizer = T5Tokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )

    # Load models with deepspeed context management
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        text_encoder = T5EncoderModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant,
            torch_dtype=weight_dtype
        )

        vae = AutoencoderKLCogVideoX.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
        )

    # transformer3d = CogVideoXTransformer3DModel.from_pretrained(
    #     args.pretrained_model_name_or_path, subfolder="transformer"
    # )
    
    transformer3d = CrossTransformer3DModel.from_pretrained(
        '/home/azhuravl/scratch/checkpoints/TrajectoryCrafter'
    )
    
    #####################
    # Keep first 16 layers
    #####################

    # Post-factum truncation
    # num_layers_to_keep = 16
    # num_frozen_layers = 8
    
    # num_cross_layers_to_keep = 2
    # num_frozen_cross_layers = 1

    # print(f"Original model has {len(transformer3d.transformer_blocks)} layers")

    # # Keep only first 16 layers
    # transformer3d.transformer_blocks = transformer3d.transformer_blocks[:num_layers_to_keep]
    # transformer3d.perceiver_cross_attention = transformer3d.perceiver_cross_attention[:num_cross_layers_to_keep]

    # print(f"Model now has {len(transformer3d.transformer_blocks)} layers and {len(transformer3d.perceiver_cross_attention)} cross-attention layers")
 
 
    ###################################################
    # Prepare LoRA model
    ###################################################

    # Freeze vae and text_encoder and set transformer3d to trainable
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    transformer3d.requires_grad_(False)


    print(transformer3d)
    
    # Lora will work with this...
    network = create_network(
        1.0,
        args.rank,
        args.network_alpha,
        text_encoder,
        transformer3d,
        neuron_dropout=None,
        add_lora_in_attn_temporal=True,
        # skip_name="norm1.linear,norm2.linear,ff.net.0.proj,ff.net.2,time_embedding,patch_embed",
        skip_name = [
            # "perceiver_cross_attention.0",
            "perceiver_cross_attention.1",
            "perceiver_cross_attention.2",
            # #
            # "transformer_blocks.0.", "transformer_blocks.1.",
            # "transformer_blocks.2.", "transformer_blocks.3.",
            # "transformer_blocks.4.", "transformer_blocks.5.",
            # "transformer_blocks.6.", "transformer_blocks.7.",
            #
            # "transformer_blocks.8.", 
            # "transformer_blocks.9.",
            # "transformer_blocks.10.", 
            "transformer_blocks.11.",
            # "transformer_blocks.12.",
            # "transformer_blocks.13.",
            "transformer_blocks.14.",
            # "transformer_blocks.15."
            # 
            "transformer_blocks.16", 
            # "transformer_blocks.17",
            # "transformer_blocks.18", 
            "transformer_blocks.19",
            # "transformer_blocks.20",
            # "transformer_blocks.21",
            "transformer_blocks.22",
            # "transformer_blocks.23",
            #
            # "transformer_blocks.24",
            "transformer_blocks.25",
            # "transformer_blocks.26",
            # "transformer_blocks.27",
            "transformer_blocks.28",
            # "transformer_blocks.29",
            # "transformer_blocks.30",
            # "transformer_blocks.31", 
            #
            "transformer_blocks.32",
            # "transformer_blocks.33",
            # "transformer_blocks.34",
            "transformer_blocks.35", 
            # "transformer_blocks.36", 
            # "transformer_blocks.37",
            "transformer_blocks.38", 
            # "transformer_blocks.39",
            # "transformer_blocks.40",
            "transformer_blocks.41",
            # 
            "norm1.linear", "norm2.linear", "ff.net.0.proj",
            "ff.net.2", "time_embedding", "patch_embed",
            "attn1.to_k", "to_out",
        ]
    )
    network.apply_to(text_encoder, transformer3d, args.train_text_encoder and not args.training_with_video_token_length, True)


    # total_params, frozen_params, trainable_params = freeze_lora_layers(
    #     network, num_frozen_layers, num_frozen_cross_layers
    # )    


    # Load custom weights if specified
    if args.transformer_path is not None:
        print(f"From checkpoint: {args.transformer_path}")
        if args.transformer_path.endswith("safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(args.transformer_path)
        else:
            state_dict = torch.load(args.transformer_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

        m, u = transformer3d.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")
        assert len(u) == 0

    if args.vae_path is not None:
        print(f"From checkpoint: {args.vae_path}")
        if args.vae_path.endswith("safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(args.vae_path)
        else:
            state_dict = torch.load(args.vae_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

        m, u = vae.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")
        assert len(u) == 0
        
        
    return tokenizer, text_encoder, vae, transformer3d, network

def setup_optimizer(args, network):
    """Setup optimizer"""
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )
        optimizer_cls = bnb.optim.AdamW8bit
    elif args.use_came:
        try:
            from came_pytorch import CAME
        except:
            raise ImportError(
                "Please install came_pytorch to use CAME. You can do so by running `pip install came_pytorch`"
            )
        optimizer_cls = CAME
    else:
        optimizer_cls = torch.optim.AdamW

    trainable_params = list(filter(lambda p: p.requires_grad, network.parameters()))
    trainable_params_optim = network.prepare_optimizer_params(args.learning_rate / 2, args.learning_rate, args.learning_rate)

    print("Number of trainable parameters:", sum(p.numel() for p in trainable_params))

    if args.use_came:
        optimizer = optimizer_cls(
            trainable_params_optim,
            lr=args.learning_rate,
            betas=(0.9, 0.999, 0.9999), 
            eps=(1e-30, 1e-16)
        )
    else:
        optimizer = optimizer_cls(
            trainable_params_optim,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    return optimizer, trainable_params


def freeze_lora_layers(network, num_frozen_layers=8, num_frozen_cross_layers=1):
    """
    Freeze specific LoRA layers based on layer indices.
    
    Args:
        network: LoRA network to freeze layers in
        num_frozen_layers: Number of transformer blocks to freeze (0 to num_frozen_layers-1)
        num_frozen_cross_layers: Number of cross-attention layers to freeze (0 to num_frozen_cross_layers-1)
    
    Returns:
        tuple: (total_params, frozen_params, trainable_params)
    """
    total_params = 0
    frozen_params = 0
    trainable_params = 0
    
    print(f"Freezing transformer layers 0-{num_frozen_layers-1} and cross-attention layers 0-{num_frozen_cross_layers-1}")
    
    for name, param in tqdm(network.named_parameters(), desc="Configuring LoRA parameters"):
        if param.requires_grad:
            should_freeze = False
            
            # Check transformer blocks (freeze layers 0 to num_frozen_layers-1)
            if 'transformer_blocks_' in name:
                parts = name.split('transformer_blocks_')
                if len(parts) > 1:
                    layer_part = parts[1].split('_')[0]
                    try:
                        layer_num = int(layer_part)
                        if layer_num < num_frozen_layers:
                            should_freeze = True
                            print(f"  Freezing transformer layer {layer_num}: {name}")
                    except ValueError:
                        pass
            
            # Check cross-attention layers (freeze layers 0 to num_frozen_cross_layers-1)
            if 'perceiver_cross_attention_' in name:
                parts = name.split('perceiver_cross_attention_')
                if len(parts) > 1:
                    layer_part = parts[1].split('_')[0]
                    try:
                        layer_num = int(layer_part)
                        if layer_num < num_frozen_cross_layers:
                            should_freeze = True
                            print(f"  Freezing cross-attention layer {layer_num}: {name}")
                    except ValueError:
                        pass
            
            if should_freeze:
                param.requires_grad = False
                frozen_params += param.numel()
            else:
                trainable_params += param.numel()
            
            total_params += param.numel()
    
    print(f"LoRA freezing complete:")
    print(f"  Total LoRA parameters: {total_params}")
    print(f"  Frozen parameters: {frozen_params}")
    print(f"  Trainable parameters: {trainable_params}")
    
    return total_params, frozen_params, trainable_params