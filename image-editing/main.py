import os
import json
from PIL import Image
import torch
from pathlib import Path
import argparse
from typing import Tuple, List, Dict, Any
import numpy as np
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image


def resize_and_pad_image(image_path: str, target_height: int = 480, target_width: int = 720) -> Image.Image:
    """
    Resize image to target height while maintaining aspect ratio, then pad or crop to target width.

    Args:
        image_path: Path to the input image
        target_height: Desired height in pixels
        target_width: Desired width in pixels

    Returns:
        Processed PIL Image
    """
    # Load image
    img = Image.open(image_path)

    # Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Calculate new width maintaining aspect ratio
    aspect_ratio = img.width / img.height
    new_width = int(target_height * aspect_ratio)

    # Resize image to target height
    img = img.resize((new_width, target_height), Image.Resampling.LANCZOS)

    # Create new black image with target dimensions
    final_img = Image.new('RGB', (target_width, target_height), (0, 0, 0))

    if new_width <= target_width:
        # Pad image
        left_padding = (target_width - new_width) // 2
        final_img.paste(img, (left_padding, 0))
    else:
        # Crop image
        left_crop = (new_width - target_width) // 2
        right_crop = left_crop + target_width
        img_cropped = img.crop((left_crop, 0, right_crop, target_height))
        final_img.paste(img_cropped, (0, 0))

    return final_img


def extract_frames(video_frames: List[Image.Image], output_dir: str, index: int) -> None:
    """
    Extract first and last frames from video and save them as images.

    Args:
        video_frames: List of PIL Images representing video frames
        output_dir: Directory to save the extracted frames
        index: Index of the current video for naming
    """
    # Extract first and last frames (already PIL Images)
    first_frame = video_frames[0]
    last_frame = video_frames[-1]

    # Save frames
    first_frame.save(os.path.join(output_dir, f'video_{index:03d}_first_frame.png'))
    last_frame.save(os.path.join(output_dir, f'video_{index:03d}_last_frame.png'))


def save_video(video_frames: List[Image.Image], output_path: str, fps: int = 8) -> None:
    """
    Convert PIL Image frames to video and save as MP4.

    Args:
        video_frames: List of PIL Images representing video frames
        output_path: Path where to save the video
        fps: Frames per second for the output video
    """
    # Convert PIL Images to numpy array with correct RGB order
    frames_np = []
    for frame in video_frames:
        # Convert PIL Image to RGB numpy array
        frame_np = np.array(frame)
        # Ensure correct color channel order (RGB)
        if frame_np.shape[-1] == 3:  # If it's a 3-channel image
            frames_np.append(frame_np[..., ::-1])  # Reverse the color channels

    frames_np = np.stack(frames_np)

    # Save as video
    export_to_video(frames_np, output_path, fps=fps)


def process_videos(annotations_path: str,
                   images_dir: str,
                   output_dir: str,
                   model_path: str = "THUDM/CogVideoX-5b-I2V") -> None:
    """
    Process images to videos using CogVideoX based on annotations.

    Args:
        annotations_path: Path to the annotations JSON file
        images_dir: Directory containing input images
        output_dir: Directory to save output videos and frames
        model_path: Path to the CogVideoX model
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load annotations
    with open(annotations_path, 'r') as f:
        annotations: List[Dict[str, str]] = json.load(f)

    # Initialize CogVideoX pipeline
    pipe = CogVideoXImageToVideoPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16
    )

    # Enable CPU offload for memory efficiency
    pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    # Process each annotation
    for idx, entry in enumerate(annotations):
        try:
            print(f"Processing video {idx + 1}/{len(annotations)}")

            # Prepare image
            image_path = os.path.join(images_dir, entry['image_name'])
            processed_image = resize_and_pad_image(image_path)

            # Generate video
            video_frames: List[Image.Image] = pipe(
                prompt=entry['video_prompt'],
                image=processed_image,
                num_inference_steps=50,
                num_frames=49,
                guidance_scale=6.0,
                generator=torch.Generator().manual_seed(42)
            ).frames[0]

            # Save video
            video_path = os.path.join(output_dir, f'video_{idx:03d}.mp4')
            save_video(video_frames, video_path)

            # Extract and save frames
            extract_frames(video_frames, output_dir, idx)

        except Exception as e:
            print(f"Error processing entry {idx}: {str(e)}")
            continue


def main() -> None:
    parser = argparse.ArgumentParser(description='Process images to videos using CogVideoX')
    parser.add_argument('--annotations', type=str, required=True,
                        help='Path to annotations JSON file')
    parser.add_argument('--images_dir', type=str, required=True,
                        help='Directory containing input images')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save output videos and frames')
    parser.add_argument('--model_path', type=str, default="THUDM/CogVideoX-5b-I2V",
                        help='Path to CogVideoX model')

    args = parser.parse_args()

    process_videos(
        annotations_path=args.annotations,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        model_path=args.model_path
    )


if __name__ == "__main__":
    main()