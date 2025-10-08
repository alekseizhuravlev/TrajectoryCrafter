from diffusers import StableDiffusionInpaintPipeline, StableDiffusionXLInpaintPipeline
import torch 
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os


if __name__ == "__main__":

    # Create output directory
    
    # get current date and time
    import datetime
    now = datetime.datetime.now()
    # format date and time as string
    date_time_str = now.strftime("%H%M%S_%d%m%Y")
    output_dir = f"/home/azhuravl/work/TrajectoryCrafter/notebooks/14_09_25/inpainting_results_{date_time_str}"
    os.makedirs(output_dir, exist_ok=True)

    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        torch_dtype=torch.float32,
    )
    # pipe = StableDiffusionInpaintPipeline.from_pretrained(
    #     "stabilityai/stable-diffusion-2-inpainting",
    #     torch_dtype=torch.float32,
    # )
    pipe.to("cuda")

    # Load your image
    # image = Image.open("/home/azhuravl/work/TrajectoryCrafter/experiments/10-09-2025/mvtracker_2to7_20250910_1459/render_basketball/frame_00001.png").convert("RGB")
    image = Image.open("/home/azhuravl/work/TrajectoryCrafter/experiments/10-09-2025/1941_mvtracker_4to5/render_1/frame_00001.png").convert("RGB")

    # Resize image to 512x512 for SD 2.1 inpainting
    image = image.resize((512, 512), Image.LANCZOS)

    # Convert to numpy array
    image_array = np.array(image)

    # Create mask for black regions
    threshold = 10  # Adjust this value if needed (0-255)

    # Create mask where black regions are white (255) and non-black regions are black (0)
    mask = np.all(image_array <= threshold, axis=2)  # True for black regions
    mask_array = mask.astype(np.uint8) * 255  # Convert to 0-255 range

    # Convert back to PIL Image and ensure it's also 512x512
    mask_image = Image.fromarray(mask_array).resize((512, 512), Image.LANCZOS)
    
    # Convert mask back to binary after resizing
    mask_array_resized = np.array(mask_image)
    mask_binary = (mask_array_resized > 127).astype(np.uint8) * 255
    mask_image = Image.fromarray(mask_binary)

    # Save original image and mask
    image.save(os.path.join(output_dir, "original_image.png"))
    mask_image.save(os.path.join(output_dir, "generated_mask.png"))

    # Create and save overlay visualization
    overlay = image_array.copy()
    overlay[mask] = [255, 0, 0]  # Red color for regions to be inpainted
    overlay_image = Image.fromarray(overlay)
    overlay_image.save(os.path.join(output_dir, "inpaint_regions_overlay.png"))

    # Visualize the original image, generated mask, and what will be inpainted
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(image)
    axs[0].set_title("Original Image")
    axs[0].axis("off")

    axs[1].imshow(mask_image, cmap='gray')
    axs[1].set_title("Generated Mask (White = Inpaint)")
    axs[1].axis("off")

    axs[2].imshow(overlay)
    axs[2].set_title("Regions to Inpaint (Red)")
    axs[2].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mask_visualization.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Mask shape: {mask_image.size}")
    print(f"Black pixels found: {np.sum(mask)} pixels")
    print(f"Percentage of image to inpaint: {(np.sum(mask) / mask.size) * 100:.1f}%")

    # Run inpainting
    prompt = (
        "A realistic scene of household objects placed on a sunlit outdoor surface: "
        "a shoe, a cereal box, a chair, and a smartphone lie naturally on the ground. "
        "The objects are complete, clearly visible, and correctly positioned with proper shadows. "
        "High-resolution and photorealistic, with no floating or distorted elements."
    )
    negative_prompt = (
        "blurry, low resolution, floating objects, distorted objects, incomplete objects, artifacts"
    )

    print("Running inpainting...")
    
    # Add additional parameters for better results
    image_inpainted = pipe(
        prompt=prompt, 
        negative_prompt=negative_prompt,
        image=image, 
        mask_image=mask_image,
        num_inference_steps=100,
        guidance_scale=5,
        generator=torch.Generator(device="cuda").manual_seed(42)
    ).images[0]

    # Save the inpainted image
    image_inpainted.save(os.path.join(output_dir, "inpainted_result.png"))

    # Create and save comparison visualization
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(image)
    axs[0].set_title("Original Image")
    axs[0].axis("off")

    axs[1].imshow(mask_image, cmap='gray')
    axs[1].set_title("Mask Image")
    axs[1].axis("off")

    axs[2].imshow(image_inpainted)
    axs[2].set_title("Inpainted Image")
    axs[2].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "inpainting_comparison.png"), dpi=300, bbox_inches='tight')
    
    print(f"All images saved to: {output_dir}")
    print("Saved files:")
    print("- original_image.png")
    print("- generated_mask.png") 
    print("- inpaint_regions_overlay.png")
    print("- mask_visualization.png")
    print("- inpainted_result.png")
    print("- inpainting_comparison.png")
    
    plt.close()