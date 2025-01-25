import argparse
import json
import os
from pathlib import Path
from typing import List
import torch
import PIL
import cv2
import numpy as np
from diffusers import CogVideoXImageToVideoPipeline


def resize_and_pad_image(image_path: str, target_height: int = 480, target_width: int = 720) -> PIL.Image.Image:
    img = PIL.Image.open(image_path).convert('RGB')
    orig_width, orig_height = img.size

    scale_ratio = min(target_width / orig_width, target_height / orig_height)
    new_width = int(orig_width * scale_ratio)
    new_height = int(orig_height * scale_ratio)

    img = img.resize((new_width, new_height), PIL.Image.Resampling.LANCZOS)
    final_img = PIL.Image.new('RGB', (target_width, target_height), (0, 0, 0))

    left_padding = (target_width - new_width) // 2
    top_padding = (target_height - new_height) // 2
    final_img.paste(img, (left_padding, top_padding))

    return final_img


def save_video(video_frames: List[PIL.Image.Image], output_path: str, fps: int = 8) -> None:
    if not video_frames:
        raise ValueError("No frames provided")

    frame = np.array(video_frames[0])
    height, width = frame.shape[:2]

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

    try:
        for frame in video_frames:
            frame_np = np.array(frame)
            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
    finally:
        out.release()


def main():
    parser = argparse.ArgumentParser(description='Generate videos using CogVideo')
    parser.add_argument('--annotations', type=str, required=True, help='Path to JSON file with image-prompt pairs')
    parser.add_argument('--images_dir', type=str, required=True, help='Directory containing input images')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save output videos')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 10, 20, 30], help='Random seeds for generation')
    parser.add_argument('--guidance_scale', type=float, default=6.0, help='Guidance scale for generation')
    parser.add_argument('--num_frames', type=int, default=49, help='Number of frames to generate')
    parser.add_argument('--num_inference_steps', type=int, default=50, help='Number of inference steps')
    args = parser.parse_args()

    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load annotations
    with open(args.annotations, 'r') as f:
        annotations = json.load(f)

    # Setup CogVideo model
    pipe = CogVideoXImageToVideoPipeline.from_pretrained("THUDM/CogVideoX-5b-I2V", torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    # Process each image-prompt pair
    for entry in annotations:
        try:
            image_name = entry['image_name']
            prompt = entry['prompt']

            # Prepare image
            image_path = os.path.join(args.images_dir, image_name)
            processed_image = resize_and_pad_image(image_path)

            # Generate videos for each seed
            for seed in args.seeds:
                seed_dir = os.path.join(args.output_dir, f"seed_{seed}")
                os.makedirs(seed_dir, exist_ok=True)

                # Generate video
                video_frames = pipe(
                    prompt=prompt,
                    image=processed_image,
                    num_frames=args.num_frames,
                    guidance_scale=args.guidance_scale,
                    num_inference_steps=args.num_inference_steps,
                    generator=torch.Generator().manual_seed(seed)
                ).frames[0]

                # Save video
                output_path = os.path.join(seed_dir, f"{Path(image_name).stem}.mp4")
                save_video(video_frames, output_path)

                print(f"Processed {image_name} with seed {seed}")

        except Exception as e:
            print(f"Error processing {image_name}: {str(e)}")
            continue


if __name__ == "__main__":
    main()