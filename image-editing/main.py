import os
import json
from PIL import Image
import torch
from pathlib import Path
import argparse
from typing import Tuple, List, Dict, Any
import numpy as np
import cv2
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import load_image
from transformers import (pipeline, set_seed,
                          LlavaNextProcessor, LlavaNextForConditionalGeneration)


def setup_llava_model() -> Tuple[LlavaNextForConditionalGeneration, LlavaNextProcessor]:
    """
    Initialize and setup the LLaVA model and processor.

    Returns:
        Tuple of (model, processor)
    """
    processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
    model = LlavaNextForConditionalGeneration.from_pretrained(
        "llava-hf/llava-v1.6-mistral-7b-hf",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    model.to("cuda:0")

    return model, processor


def get_image_caption(image: Image.Image,
                      model: LlavaNextForConditionalGeneration,
                      processor: LlavaNextProcessor,
                      prompt: str = "Describe this image in detail.") -> str:
    """
    Generate a caption for an image using LLaVA.

    Args:
        image: PIL Image to caption
        model: LLaVA model
        processor: LLaVA processor
        prompt: Text prompt for image captioning

    Returns:
        Generated caption
    """
    # Prepare conversation prompt
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image"},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    # Process inputs
    inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda:0")

    # Generate caption
    output = model.generate(**inputs, max_new_tokens=100)
    raw_caption = processor.decode(output[0], skip_special_tokens=True)

    # Clean up the caption - remove everything before and including [/INST]
    if "[/INST]" in raw_caption:
        cleaned_caption = raw_caption.split("[/INST]")[1].strip()
    else:
        cleaned_caption = raw_caption.strip()

    return cleaned_caption


def setup_llama_pipeline(llama_seed: int | None) -> Any:
    """
    Initialize and setup the Llama 3 pipeline.

    Args:
        llama_seed: Optional seed for Llama generation. If provided, sets global seed.

    Returns:
        The configured pipeline
    """
    # Set seed if provided
    if llama_seed is not None:
        set_seed(llama_seed)

    # Initialize the pipeline
    pipe = pipeline(
        "text-generation",
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        model_kwargs={"torch_dtype": torch.bfloat16},
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Set up the tokenizer
    pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id

    return pipe


def get_video_prompt(llama_pipe: Any,
                     original_prompt: str,
                     role: str,
                     max_new_tokens: int,
                     temperature: float,
                     top_p: float,
                     do_sample: bool) -> str:
    """
    Generate a video prompt from an image prompt using Llama 3.

    Args:
        llama_pipe: The Llama 3 pipeline
        original_prompt: The original image prompt
        role: The system role/instruction
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        do_sample: Whether to use sampling

    Returns:
        The generated video prompt
    """
    # Format the message as a conversation
    message = [
        {"role": "system", "content": role},
        {"role": "user", "content": original_prompt}
    ]

    # Generate text
    output = llama_pipe(
        message,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
    )

    # Extract the generated prompt
    return output[0]["generated_text"][-1]["content"].strip()


def resize_and_pad_image(image_path: str, target_height: int = 480, target_width: int = 720) -> Image.Image:
    """
    Resize image (up or down) such that its larger dimension fits the required size, then pad the other dimension.
    Handles both upsampling and downsampling cases.

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

    # Get original dimensions
    orig_width, orig_height = img.size

    # Calculate scaling ratios relative to targets
    width_ratio = target_width / orig_width
    height_ratio = target_height / orig_height

    # For upsampling (image smaller than target):
    # If both dimensions are smaller, use the larger ratio to ensure smallest dimension meets target
    # For downsampling (image larger than target):
    # Use the smaller ratio to ensure largest dimension meets target
    if orig_width <= target_width and orig_height <= target_height:
        # Upsampling case - use larger ratio
        scale_ratio = max(width_ratio, height_ratio)
    else:
        # Downsampling case - use smaller ratio
        scale_ratio = min(width_ratio, height_ratio)

    # Calculate new dimensions
    new_width = int(orig_width * scale_ratio)
    new_height = int(orig_height * scale_ratio)

    # Resize image
    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Create new black image with target dimensions
    final_img = Image.new('RGB', (target_width, target_height), (0, 0, 0))

    # Calculate padding
    left_padding = (target_width - new_width) // 2
    top_padding = (target_height - new_height) // 2

    # Paste resized image onto black background
    final_img.paste(img, (left_padding, top_padding))

    return final_img


def extract_frames(video_frames: List[Image.Image], output_path: str) -> None:
    """
    Extract first and last frames from video and save them as images.

    Args:
        video_frames: List of PIL Images representing video frames
        output_path: Base path for saving the frames (without extension)
    """
    # Extract first and last frames (already PIL Images)
    first_frame = video_frames[0]
    last_frame = video_frames[-1]

    # Save frames
    first_frame.save(f"{output_path}_first_frame.png")
    last_frame.save(f"{output_path}_last_frame.png")


def save_video(video_frames: List[Image.Image], output_path: str, fps: int = 8) -> None:
    """
    Convert PIL Image frames to video and save as MP4 using OpenCV.

    Args:
        video_frames: List of PIL Images representing video frames
        output_path: Path where to save the video
        fps: Frames per second for the output video
    """
    if not video_frames:
        raise ValueError("No frames provided")

    # Convert first frame to numpy array to get dimensions
    frame = np.array(video_frames[0])
    height, width = frame.shape[:2]

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )

    try:
        # Write each frame
        for frame in video_frames:
            # Convert PIL Image to numpy array and change color space from RGB to BGR
            frame_np = np.array(frame)
            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
    finally:
        # Make sure to release the VideoWriter
        out.release()


def process_videos(annotations_path: str,
                   images_dir: str,
                   output_dir: str,
                   seeds: List[int],
                   guidance_scales: List[float],
                   max_new_tokens: int,
                   temperature: float,
                   top_p: float,
                   do_sample: bool,
                   llama_seed: int | None,
                   model_path: str = "THUDM/CogVideoX-5b-I2V") -> None:
    """
    Process images to videos using CogVideoX based on annotations.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load annotations
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)

    # Setup models
    llama_pipe = setup_llama_pipeline(llama_seed)
    llava_model, llava_processor = setup_llava_model()
    cog_pipe = CogVideoXImageToVideoPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16
    )

    # Enable CPU offload for memory efficiency
    cog_pipe.enable_sequential_cpu_offload()
    cog_pipe.vae.enable_slicing()
    cog_pipe.vae.enable_tiling()

    # Get settings and images from annotations
    settings = annotations.get("settings", [])
    images = annotations.get("images", [])

    def replace_placeholders(text: str, context: Dict[str, str]) -> str:
        """Replace placeholders in text with their values from context."""
        for key, value in context.items():
            placeholder = f"${{{key}}}"
            if placeholder in text:
                text = text.replace(placeholder, value)
        return text

    # Process each setting
    for setting_idx, setting in enumerate(settings):
        print(f"Processing setting {setting_idx + 1}/{len(settings)}")

        # Process each image with current setting
        for idx, entry in enumerate(images):
            try:
                print(f"  Processing image {idx + 1}/{len(images)}")

                # Get base name from image_name (remove extension)
                base_name = os.path.splitext(entry['image_name'])[0]

                # Initialize context for placeholder replacement
                context = {
                    "image_prompt": entry['image_prompt']
                }

                # Run LLaVA if prompt is provided in settings
                if "llava_prompt" in setting:
                    # Prepare image
                    image_path = os.path.join(images_dir, entry['image_name'])
                    image = Image.open(image_path)
                    if image.mode != 'RGB':
                        image = image.convert('RGB')

                    # Get LLaVA caption
                    llava_caption = get_image_caption(
                        image=image,
                        model=llava_model,
                        processor=llava_processor,
                        prompt=setting["llava_prompt"]
                    )
                    print(f"    LLaVA caption: {llava_caption}")
                    context["llava_output"] = llava_caption

                # Process with Llama
                llama_prompt = replace_placeholders(setting["llama_prompt"], context)

                # Generate video prompt using Llama
                llama_output = get_video_prompt(
                    llama_pipe=llama_pipe,
                    original_prompt=llama_prompt,
                    role=setting["llama_role"],
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample
                )
                print(f"    Llama output: {llama_output}")
                context["llama_output"] = llama_output

                # Get final prompt for CogVideo
                cog_prompt = replace_placeholders(setting["cog_prompt"], context)
                print(f"    Final prompt: {cog_prompt}")

                # Prepare image for CogVideoX if not done already
                if "processed_image" not in context:
                    image_path = os.path.join(images_dir, entry['image_name'])
                    processed_image = resize_and_pad_image(image_path)
                    context["processed_image"] = processed_image

                # Generate videos with different seeds and guidance scales
                variant_count = 0
                for seed in seeds:
                    for guidance_scale in guidance_scales:
                        variant_count += 1
                        print(f"      Generating variant {variant_count}/{len(seeds) * len(guidance_scales)} "
                              f"(seed={seed}, guidance_scale={guidance_scale})")

                        # Generate video
                        video_frames: List[Image.Image] = cog_pipe(
                            prompt=cog_prompt,
                            image=context["processed_image"],
                            num_inference_steps=50,
                            num_frames=49,
                            guidance_scale=guidance_scale,
                            generator=torch.Generator().manual_seed(seed)
                        ).frames[0]

                        # Create base path for this variant
                        base_path = os.path.join(
                            output_dir,
                            f'{base_name}_setting_{setting_idx}_seed_{seed}_guidance_{guidance_scale:.1f}'
                        )

                        # Save video
                        save_video(video_frames, f"{base_path}.mp4")

                        # Extract and save frames
                        extract_frames(video_frames, base_path)

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
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456],
                        help='Random seeds to use for generation')
    parser.add_argument('--guidance_scales', type=float, nargs='+', default=[6.0, 7.0, 8.0],
                        help='Guidance scale values to use for generation')
    parser.add_argument('--max_new_tokens', type=int, default=512,
                        help='Maximum number of tokens for Llama generation')
    parser.add_argument('--temperature', type=float, default=0.6,
                        help='Temperature for Llama generation sampling')
    parser.add_argument('--top_p', type=float, default=0.9,
                        help='Top-p (nucleus sampling) parameter for Llama generation')
    parser.add_argument('--no_sample', action='store_false', dest='do_sample',
                        help='Disable sampling in Llama generation (use greedy decoding)')
    parser.add_argument('--llama_seed', type=int, default=None,
                        help='Seed for Llama text generation (optional)')

    parser.set_defaults(do_sample=True)

    args = parser.parse_args()

    process_videos(
        annotations_path=args.annotations,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        seeds=args.seeds,
        guidance_scales=args.guidance_scales,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.do_sample,
        llama_seed=args.llama_seed,
        model_path=args.model_path
    )


if __name__ == "__main__":
    main()