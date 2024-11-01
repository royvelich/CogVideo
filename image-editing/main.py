import os
import json
from PIL import Image
import torch
from pathlib import Path
import argparse
from typing import Tuple, List, Dict, Any
import numpy as np
import cv2
from datetime import datetime
from typing import NamedTuple
import dataclasses
import copy
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import load_image
from transformers import (pipeline, set_seed,
                          LlavaNextProcessor, LlavaNextForConditionalGeneration)


@dataclasses.dataclass
class ImageProcessingLog:
    """Class to store processing information for each image."""
    image_name: str
    image_prompt: str
    llava_prompt: str | None = None
    llava_output: str | None = None
    final_prompt: str | None = None

    def __setattr__(self, name: str, value: Any) -> None:
        """Allow dynamic addition of Llama-specific attributes."""
        if not hasattr(self, name):
            # Create new field dynamically
            self.__dict__[name] = value
            # Add field to dataclass fields
            field = dataclasses.field(default=None)
            self.__class__.__dataclass_fields__[name] = field
        else:
            super().__setattr__(name, value)


def split_into_groups(items: List[Any], num_groups: int) -> List[List[Any]]:
    """
    Split a list into K approximately equal sized groups.

    Args:
        items: List to split
        num_groups: Number of groups to create

    Returns:
        List of groups
    """
    # Calculate base size and remainder
    n = len(items)
    base_size = n // num_groups
    remainder = n % num_groups

    groups = []
    start = 0

    # Create groups
    for i in range(num_groups):
        # Add one extra item to the first 'remainder' groups
        group_size = base_size + (1 if i < remainder else 0)
        end = start + group_size
        groups.append(items[start:end])
        start = end

    return groups


def setup_run_directories(base_output_dir: str) -> str:
    """
    Create directory structure for current run.

    Args:
        base_output_dir: Base output directory path

    Returns:
        Run directory path
    """
    # Create run directory with current date
    current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_output_dir, current_date)
    os.makedirs(run_dir, exist_ok=True)

    return run_dir


def save_run_config(run_dir: str, args: argparse.Namespace) -> None:
    """
    Save run configuration to JSON file.

    Args:
        run_dir: Run directory path
        args: Command line arguments
    """
    config = {k: v for k, v in vars(args).items()}
    config_path = os.path.join(run_dir, "run_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


def update_processing_log(log_path: str, image_log: ImageProcessingLog) -> None:
    """
    Update the processing log with new image information.

    Args:
        log_path: Path to log file
        image_log: Image processing information
    """
    # Load existing log if it exists
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            log_data = json.load(f)
    else:
        log_data = {}

    # Update log with new image data
    log_data[image_log.image_name] = dataclasses.asdict(image_log)

    # Save updated log
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2)


def setup_llava_model() -> Tuple[LlavaNextForConditionalGeneration, LlavaNextProcessor]:
    """
    Initialize and setup the LLaVA model and processor.

    Returns:
        Tuple of (model, processor)
    """
    processor = LlavaNextProcessor.from_pretrained("llava-v1.6-vicuna-7b-hf")
    model = LlavaNextForConditionalGeneration.from_pretrained(
        "llava-v1.6-vicuna-7b-hf",
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
    caption = processor.decode(output[0], skip_special_tokens=True)

    # Clean up caption - remove everything before and including [/INST]
    if "[/INST]" in caption:
        caption = caption.split("[/INST]")[1].strip()

    return caption


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
                   model_path: str = "THUDM/CogVideoX-5b-I2V",
                   num_groups: int | None = None,
                   group_index: int | None = None) -> None:
    """
    Process images to videos using CogVideoX based on annotations.
    """
    # Create run directory structure
    run_dir = setup_run_directories(output_dir)

    # Load annotations
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)

    # Setup models only if needed
    all_settings = annotations.get("settings", [])

    # Check if we need Llama and LLaVA models
    needs_llama = any(any(key.startswith("llama") for key in setting.keys()) for setting in all_settings)
    needs_llava = any("llava" in setting for setting in all_settings)

    llama_pipe = None
    llava_model, llava_processor = None, None

    if needs_llama:
        llama_pipe = setup_llama_pipeline(llama_seed)
    if needs_llava:
        llava_model, llava_processor = setup_llava_model()

    # Setup CogVideo model
    cog_pipe = CogVideoXImageToVideoPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16
    )

    cog_pipe.enable_sequential_cpu_offload()
    cog_pipe.vae.enable_slicing()
    cog_pipe.vae.enable_tiling()

    # Handle group processing
    all_images = annotations.get("images", [])
    if num_groups is not None and group_index is not None:
        image_groups = split_into_groups(all_images, num_groups)
        if not 0 <= group_index < num_groups:
            raise ValueError(f"Group index {group_index} is out of range [0, {num_groups - 1}]")
        images = image_groups[group_index]
        print(f"Processing group {group_index + 1}/{num_groups} with {len(images)} images")
    else:
        images = all_images
        image_groups = [all_images]

    # Save group information
    group_info = {
        "total_images": len(all_images),
        "num_groups": num_groups,
        "group_index": group_index,
        "group_size": len(images),
        "group_start_idx": sum(len(g) for g in image_groups[:group_index]) if num_groups is not None else 0,
        "group_end_idx": sum(len(g) for g in image_groups[:group_index + 1]) if num_groups is not None else len(all_images)
    }
    with open(os.path.join(run_dir, "group_info.json"), 'w') as f:
        json.dump(group_info, f, indent=2)

    def replace_placeholders(text: str, context: Dict[str, str]) -> str:
        """Replace placeholders in text with their values from context."""
        result = text
        for key, value in sorted(context.items(), key=lambda x: len(x[0]), reverse=True):
            placeholder = f"${{{key}}}"
            if placeholder in result:
                result = result.replace(placeholder, value)
        return result

    # Process each setting
    for setting_idx, setting in enumerate(all_settings):
        print(f"Processing setting {setting_idx + 1}/{len(all_settings)}")

        setting_dir = os.path.join(run_dir, f"setting_{setting_idx}")
        os.makedirs(setting_dir, exist_ok=True)
        log_path = os.path.join(setting_dir, "processing_log.json")

        # Process each image
        for idx, entry in enumerate(images):
            try:
                print(f"  Processing image {idx + 1}/{len(images)}")

                image_log = ImageProcessingLog(
                    image_name=entry['image_name'],
                    image_prompt=entry['image_prompt']
                )

                base_name = os.path.splitext(entry['image_name'])[0]
                context = {
                    "image_prompt": entry['image_prompt']
                }

                final_prompt = None

                # Run LLaVA if specified in settings
                if "llava" in setting and llava_model is not None:
                    llava_config = setting["llava"]
                    image_log.llava_prompt = llava_config["prompt"]

                    image_path = os.path.join(images_dir, entry['image_name'])
                    image = Image.open(image_path)
                    if image.mode != 'RGB':
                        image = image.convert('RGB')

                    llava_caption = get_image_caption(
                        image=image,
                        model=llava_model,
                        processor=llava_processor,
                        prompt=llava_config["prompt"]
                    )
                    print(f"    LLaVA caption: {llava_caption}")
                    context["llava_output"] = llava_caption
                    image_log.llava_output = llava_caption

                # Process with multiple Llama runs if specified in settings
                llama_keys = sorted([k for k in setting.keys() if k.startswith("llama")],
                                    key=lambda x: int(x[5:]) if x[5:].isdigit() else float('inf'))

                for llama_key in llama_keys:
                    if llama_pipe is not None:
                        config = setting[llama_key]
                        print(f"    Running {llama_key}")

                        llama_prompt = replace_placeholders(config["prompt"], context)

                        # Store prompt in log with the specific index
                        setattr(image_log, f"{llama_key}_prompt", llama_prompt)
                        setattr(image_log, f"{llama_key}_role", config["role"])

                        llama_output = get_video_prompt(
                            llama_pipe=llama_pipe,
                            original_prompt=llama_prompt,
                            role=config["role"],
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            do_sample=do_sample
                        )
                        print(f"    {llama_key} output: {llama_output}")

                        # Store output in context with specific key
                        context[f"{llama_key}_output"] = llama_output
                        # Store output in log with specific key
                        setattr(image_log, f"{llama_key}_output", llama_output)

                # Get final prompt for CogVideo
                if "cog" not in setting:
                    raise ValueError(f"Setting {setting_idx} missing required 'cog' configuration")

                final_prompt = replace_placeholders(setting["cog"]["prompt"], context)
                print(f"    Final prompt: {final_prompt}")
                image_log.final_prompt = final_prompt

                # Update processing log
                update_processing_log(log_path, image_log)

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

                        video_frames: List[Image.Image] = cog_pipe(
                            prompt=final_prompt,
                            image=context["processed_image"],
                            num_inference_steps=50,
                            num_frames=49,
                            guidance_scale=guidance_scale,
                            generator=torch.Generator().manual_seed(seed)
                        ).frames[0]

                        base_path = os.path.join(
                            setting_dir,
                            f'{base_name}_seed_{seed}_guidance_{guidance_scale:.1f}'
                        )

                        save_video(video_frames, f"{base_path}.mp4")
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
    parser.add_argument('--num_groups', type=int, default=None,
                        help='Number of groups to split images into')
    parser.add_argument('--group_index', type=int, default=None,
                        help='Index of group to process (0-based)')

    parser.set_defaults(do_sample=True)

    args = parser.parse_args()

    # Validate group processing arguments
    if (args.num_groups is None) != (args.group_index is None):
        parser.error("Both --num_groups and --group_index must be provided for group processing")

    if args.num_groups is not None and args.group_index is not None:
        if args.group_index >= args.num_groups:
            parser.error(f"Group index must be less than number of groups (0 to {args.num_groups - 1})")

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
        model_path=args.model_path,
        num_groups=args.num_groups,
        group_index=args.group_index
    )


if __name__ == "__main__":
    main()