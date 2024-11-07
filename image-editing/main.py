import os
import json
from PIL import Image, ImageFile
import torch
from pathlib import Path
import argparse
from typing import Tuple, List, Dict, Any
import numpy as np
import cv2
from datetime import datetime
import sys
import shutil
import paramiko
import re
import socket
from typing import NamedTuple
import dataclasses
import copy
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import load_image
from transformers import (pipeline, set_seed,
                          LlavaNextProcessor, LlavaNextForConditionalGeneration)
ImageFile.LOAD_TRUNCATED_IMAGES = True


@dataclasses.dataclass
class ImageProcessingLog:
    """Class to store processing information for each image."""
    image_name: str
    video_prompts: List[str]
    prompt_index: int
    llava_prompt: str | None = None
    llava_output: str | None = None
    final_prompt: str | None = None

    def __setattr__(self, name: str, value: Any) -> None:
        """Allow dynamic addition of Llama-specific attributes."""
        if not hasattr(self, name):
            self.__dict__[name] = value
            field = dataclasses.field(default=None)
            self.__class__.__dataclass_fields__[name] = field
        else:
            super().__setattr__(name, value)


def setup_run_directories(base_output_dir: str) -> str:
    """
    Create directory structure for current run.

    Args:
        base_output_dir: Base output directory path

    Returns:
        Run directory path
    """
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
    config = {
        "command_line_args": vars(args),
        "execution_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "python_command": f"python {' '.join(sys.argv)}",
        "environment": {
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
        }
    }

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

    # Create a dictionary directly from all dataclass fields
    log_entry = {
        field_name: getattr(image_log, field_name)
        for field_name in image_log.__dataclass_fields__
    }

    # Create unique key combining image name and prompt index
    log_key = f"{image_log.image_name}_prompt_{image_log.prompt_index}"

    # Update log with new image data
    log_data[log_key] = log_entry

    # Save updated log
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2)


def split_into_groups(items: List[Any], num_groups: int) -> List[List[Any]]:
    """
    Split a list into K approximately equal sized groups.

    Args:
        items: List to split
        num_groups: Number of groups to create

    Returns:
        List of groups
    """
    n = len(items)
    base_size = n // num_groups
    remainder = n % num_groups

    groups = []
    start = 0

    for i in range(num_groups):
        group_size = base_size + (1 if i < remainder else 0)
        end = start + group_size
        groups.append(items[start:end])
        start = end

    return groups


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

    inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda:0")

    output = model.generate(**inputs, max_new_tokens=100)
    caption = processor.decode(output[0], skip_special_tokens=True)

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
    if llama_seed is not None:
        set_seed(llama_seed)

    pipe = pipeline(
        "text-generation",
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        model_kwargs={"torch_dtype": torch.bfloat16},
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

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
    message = [
        {"role": "system", "content": role},
        {"role": "user", "content": original_prompt}
    ]

    output = llama_pipe(
        message,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
    )

    return output[0]["generated_text"][-1]["content"].strip()


def resize_and_pad_image(image_path: str, target_height: int = 480, target_width: int = 720) -> Image.Image:
    """
    Resize image (up or down) such that its larger dimension fits the required size, then pad the other dimension.

    Args:
        image_path: Path to the input image
        target_height: Desired height in pixels
        target_width: Desired width in pixels

    Returns:
        Processed PIL Image
    """
    img = Image.open(image_path)

    if img.mode != 'RGB':
        img = img.convert('RGB')

    orig_width, orig_height = img.size

    width_ratio = target_width / orig_width
    height_ratio = target_height / orig_height

    if orig_width <= target_width and orig_height <= target_height:
        scale_ratio = max(width_ratio, height_ratio)
    else:
        scale_ratio = min(width_ratio, height_ratio)

    new_width = int(orig_width * scale_ratio)
    new_height = int(orig_height * scale_ratio)

    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    final_img = Image.new('RGB', (target_width, target_height), (0, 0, 0))

    left_padding = (target_width - new_width) // 2
    top_padding = (target_height - new_height) // 2

    final_img.paste(img, (left_padding, top_padding))

    return final_img


def extract_frames(video_frames: List[Image.Image], output_path: str) -> None:
    """
    Extract first and last frames from video and save them as images.

    Args:
        video_frames: List of PIL Images representing video frames
        output_path: Base path for saving the frames (without extension)
    """
    first_frame = video_frames[0]
    last_frame = video_frames[-1]

    first_frame.save(f"{output_path}_first_frame.png")
    last_frame.save(f"{output_path}_last_frame.png")


def save_video(video_frames: List[Image.Image], output_path: str, fps: int = 8, video_format: str = 'mp4') -> None:
    """
    Convert PIL Image frames to video and save using OpenCV in specified format.

    Args:
        video_frames: List of PIL Images representing video frames
        output_path: Path where to save the video
        fps: Frames per second for the output video
        video_format: Format of the output video (e.g., 'mp4', 'avi')
    """
    if not video_frames:
        raise ValueError("No frames provided")

    frame = np.array(video_frames[0])
    height, width = frame.shape[:2]

    FOURCC_CODES = {
        'mp4': cv2.VideoWriter_fourcc(*'mp4v'),
        'avi': cv2.VideoWriter_fourcc(*'XVID'),
        'mov': cv2.VideoWriter_fourcc(*'mp4v'),
        # Add more formats if needed
    }
    
    fourcc = FOURCC_CODES.get(video_format.lower())
    if fourcc is None:
        raise ValueError(f"Unsupported video format: {video_format}")

    out = cv2.VideoWriter(
        output_path,
        fourcc,
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


def remote_file_exists(sftp_client: paramiko.SFTPClient, remote_path: str) -> bool:
    try:
        sftp_client.stat(remote_path)
        return True
    except FileNotFoundError:
        return False
    except IOError:
        return False  # For older versions of Paramiko


def copy_file_to_remote(sftp_client: paramiko.SFTPClient, local_path: str, remote_path: str) -> None:
    sftp_client.put(local_path, remote_path)


def sftp_mkdirs(sftp: paramiko.SFTPClient, remote_directory: str) -> None:
    dirs = []
    while remote_directory not in ('/', ''):
        dirs.append(remote_directory)
        remote_directory, _ = os.path.split(remote_directory)
    dirs.reverse()
    for directory in dirs:
        try:
            sftp.stat(directory)
        except FileNotFoundError:
            sftp.mkdir(directory)


def process_videos(annotations_path: str,
                   images_dir: str,
                   run_dir: str,
                   seeds: List[int],
                   guidance_scales: List[float],
                   max_new_tokens: int,
                   temperature: float,
                   top_p: float,
                   do_sample: bool,
                   llama_seed: int | None,
                   num_inference_steps: int,
                   num_frames: int,
                   model_path: str = "THUDM/CogVideoX-5b-I2V",
                   num_groups: int | None = None,
                   group_index: int | None = None,
                   video_format: str = 'mp4',
                   secondary_output_dir: str | None = None,
                   user: str | None = None,
                   host: str | None = None,
                   ssh_password: str | None = None) -> None:
    """
    Process images to videos using CogVideoX based on annotations.
    """
    ssh = None
    sftp = None

    # Load annotations
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)

    # Setup models only if needed
    all_settings = annotations.get("settings", [])
    all_images = annotations.get("images", [])

    # Find the most common number of prompts across all images
    prompt_counts = {}
    for img in all_images:
        num = len(img['video_prompts'])
        prompt_counts[num] = prompt_counts.get(num, 0) + 1

    num_prompts = max(prompt_counts.items(), key=lambda x: x[1])[0]
    print(f"Most common number of prompts per image: {num_prompts} (occurs in {prompt_counts[num_prompts]} images)")
    print(f"Number of prompts distribution: {prompt_counts}")

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

    remote_base_dir = secondary_output_dir
    # Setup secondary output directory if provided
    if secondary_output_dir:
        if os.path.exists(secondary_output_dir):
            is_remote = False
            print(f"Secondary output directory exists locally: {secondary_output_dir}")
        else:
            is_remote = True
            if not host or not user:
                raise ValueError("Host and user must be provided for remote connections.")

            # Set up SSH and SFTP clients
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

            # Attempt to connect
            try:
                if ssh_password:
                    ssh.connect(hostname=host, username=user, password=ssh_password)
                else:
                    ssh.connect(hostname=host, username=user)
            except paramiko.AuthenticationException:
                import getpass
                ssh_password = getpass.getpass(f"Password for {user}@{host}: ")
                ssh.connect(hostname=host, username=user, password=ssh_password)

            sftp = ssh.open_sftp()
            remote_base_dir = secondary_output_dir
            print(f"Connected to remote host {host} as user {user}")
    else:
        remote_base_dir = None
        is_remote = False
        raise ValueError("Secondary output directory not provided")
    

    # Process each setting
    for setting_idx, setting in enumerate(all_settings):
        print(f"Processing setting {setting_idx + 1}/{len(all_settings)}")
        setting_dir = os.path.join(run_dir, f"setting_{setting_idx}")
        os.makedirs(setting_dir, exist_ok=True)
        log_path = os.path.join(setting_dir, "processing_log.json")

        # Loops: seeds -> prompt_indices -> guidance_scales -> images
        for seed in seeds:
            print(f"Processing seed {seed}")

            for prompt_idx in range(num_prompts):
                print(f"  Processing prompt index {prompt_idx + 1}/{num_prompts}")

                # Filter images that have enough prompts for this prompt_idx
                valid_images = [img for img in images if len(img['video_prompts']) > prompt_idx]
                if not valid_images:
                    print(f"    No images have {prompt_idx + 1} prompts, skipping this prompt index")
                    continue

                print(f"    Found {len(valid_images)} images with prompt index {prompt_idx}")

                for guidance_scale in guidance_scales:
                    print(f"      Processing guidance scale {guidance_scale}")

                    for idx, entry in enumerate(valid_images):
                        try:
                            print(f"        Processing image {idx + 1}/{len(valid_images)}")
                            base_name = os.path.splitext(entry['image_name'])[0]
                            video_prompt = entry['video_prompts'][prompt_idx]

                            image_log = ImageProcessingLog(
                                image_name=entry['image_name'],
                                video_prompts=entry['video_prompts'],
                                prompt_index=prompt_idx
                            )

                            context = {
                                "video_prompt": video_prompt
                            }

                            # Run LLaVA if specified
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
                                print(f"          LLaVA caption: {llava_caption}")
                                context["llava_output"] = llava_caption
                                image_log.llava_output = llava_caption

                            # Process with multiple Llama runs
                            llama_keys = sorted([k for k in setting.keys() if k.startswith("llama")],
                                                key=lambda x: int(x[5:]) if x[5:].isdigit() else float('inf'))

                            for llama_key in llama_keys:
                                if llama_pipe is not None:
                                    config = setting[llama_key]
                                    print(f"          Running {llama_key}")

                                    llama_prompt = replace_placeholders(config["prompt"], context)
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
                                    print(f"          {llama_key} output: {llama_output}")

                                    context[f"{llama_key}_output"] = llama_output
                                    setattr(image_log, f"{llama_key}_output", llama_output)

                            # Get final prompt for CogVideo
                            if "cog" not in setting:
                                raise ValueError(f"Setting {setting_idx} missing required 'cog' configuration")

                            final_prompt = replace_placeholders(setting["cog"]["prompt"], context)
                            print(f"          Final prompt: {final_prompt}")
                            image_log.final_prompt = final_prompt

                            # Update processing log
                            update_processing_log(log_path, image_log)

                            # Prepare image for CogVideoX
                            if "processed_image" not in context:
                                image_path = os.path.join(images_dir, entry['image_name'])
                                processed_image = resize_and_pad_image(image_path)
                                context["processed_image"] = processed_image

                            # Generate base path
                            base_path = os.path.join(
                                setting_dir,
                                f'{base_name}_prompt_{prompt_idx}_seed_{seed}_guidance_{guidance_scale:.1f}'
                            )
                            video_file = f"{base_path}.{video_format}"
                            first_frame_file = f"{base_path}_first_frame.png"
                            last_frame_file = f"{base_path}_last_frame.png"

                            # Before processing, check if files exist in secondary output dir
                            skip_processing = False
                            if secondary_output_dir:
                                # Construct remote paths
                                relative_path = os.path.relpath(video_file, run_dir)
                                remote_video_file = os.path.join(remote_base_dir, relative_path)
                                relative_first_frame = os.path.relpath(first_frame_file, run_dir)
                                remote_first_frame_file = os.path.join(remote_base_dir, relative_first_frame)
                                relative_last_frame = os.path.relpath(last_frame_file, run_dir)
                                remote_last_frame_file = os.path.join(remote_base_dir, relative_last_frame)

                                # Check if files exist
                                if is_remote:
                                    # Use sftp to check
                                    files_exist = all(remote_file_exists(sftp, remote_file) for remote_file in
                                                      [remote_video_file, remote_first_frame_file, remote_last_frame_file])
                                else:
                                    # Local path
                                    files_exist = all(os.path.exists(remote_file) for remote_file in
                                                      [remote_video_file, remote_first_frame_file, remote_last_frame_file])

                                if files_exist:
                                    print(f"          Files already exist in secondary directory, skipping processing.")
                                    continue

                            # Generate video
                            video_frames: List[Image.Image] = cog_pipe(
                                prompt=final_prompt,
                                image=context["processed_image"],
                                num_inference_steps=num_inference_steps,
                                num_frames=num_frames,
                                guidance_scale=guidance_scale,
                                generator=torch.Generator().manual_seed(seed)
                            ).frames[0]

                            save_video(video_frames, video_file, video_format=video_format)
                            extract_frames(video_frames, base_path)

                            # After processing, copy files to secondary output dir
                            if secondary_output_dir:
                                # Ensure remote directories exist
                                if is_remote:
                                    remote_dir = os.path.dirname(remote_video_file)
                                    sftp_mkdirs(sftp, remote_dir)
                                else:
                                    os.makedirs(os.path.dirname(remote_video_file), exist_ok=True)

                                # Copy files
                                if is_remote:
                                    copy_file_to_remote(sftp, video_file, remote_video_file)
                                    copy_file_to_remote(sftp, first_frame_file, remote_first_frame_file)
                                    copy_file_to_remote(sftp, last_frame_file, remote_last_frame_file)
                                else:
                                    shutil.copyfile(video_file, remote_video_file)
                                    shutil.copyfile(first_frame_file, remote_first_frame_file)
                                    shutil.copyfile(last_frame_file, remote_last_frame_file)

                        except Exception as e:
                            print(f"Error processing entry {idx} ({entry['image_name']}): {str(e)}")
                            continue

    # Close SSH and SFTP connections if any
    if secondary_output_dir and 'ssh' in locals() and is_remote:
        sftp.close()
        ssh.close()


def main() -> None:
    parser = argparse.ArgumentParser(description='Process images to videos using CogVideoX')
    # Previous arguments remain the same
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
    parser.add_argument('--num_inference_steps', type=int, default=50,
                        help='Number of inference steps for video generation')
    parser.add_argument('--num_frames', type=int, default=49,
                        help='Number of frames to generate')
    parser.add_argument('--video_format', type=str, default='avi',
                        help='Video format to save the output videos (e.g., mp4, avi)')
    parser.add_argument('--secondary_output_dir', type=str, default="/mnt/gipnetapp_public/video_edit/results",
                        help='Secondary directory to save output videos and frames (can be remote)')

    parser.add_argument('--user', type=str, default="snoamr")
    parser.add_argument('--host', type=str, default="132.68.39.112")
    parser.add_argument('--ssh_password', type=str, default="keypurNR9294!")


    parser.set_defaults(do_sample=True)

    args = parser.parse_args()

    # Validation code remains the same
    if (args.num_groups is None) != (args.group_index is None):
        parser.error("Both --num_groups and --group_index must be provided for group processing")

    if args.num_groups is not None and args.group_index is not None:
        if args.group_index >= args.num_groups:
            parser.error(f"Group index must be less than number of groups (0 to {args.num_groups - 1})")

    # Create run directory
    run_dir = setup_run_directories(args.output_dir)

    # Save run configuration
    save_run_config(run_dir, args)

    process_videos(
        annotations_path=args.annotations,
        images_dir=args.images_dir,
        run_dir=run_dir,
        seeds=args.seeds,
        guidance_scales=args.guidance_scales,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.do_sample,
        llama_seed=args.llama_seed,
        num_inference_steps=args.num_inference_steps,
        num_frames=args.num_frames,
        model_path=args.model_path,
        num_groups=args.num_groups,
        group_index=args.group_index,
        video_format=args.video_format,
        secondary_output_dir=args.secondary_output_dir,
        user=args.user,
        host=args.host,
        ssh_password=args.ssh_password
    )


if __name__ == "__main__":
    main()
