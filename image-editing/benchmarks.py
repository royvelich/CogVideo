import argparse
import json
import os
from pathlib import Path
import PIL
import torch
from tqdm import tqdm
import numpy as np
from torch import autocast, inference_mode
from diffusers import (
    StableDiffusionImg2ImgPipeline,
    StableDiffusionPipeline,
    DDIMScheduler,
    UNet2DConditionModel
)
from leditspp.scheduling_dpmsolver_multistep_inject import DPMSolverMultistepSchedulerInject
from leditspp import StableDiffusionPipeline_LEDITS
from transformers import CLIPTextModel, CLIPTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description='Process images using LEDITS++, SD img2img, or DDPM')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing input images')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory for output images')
    parser.add_argument('--json_path', type=str, required=True, help='Path to JSON file with prompts')
    parser.add_argument('--pipeline', type=str, choices=['ledits', 'img2img', 'ddpm'], required=True,
                        help='Which pipeline to use')
    parser.add_argument('--seeds', type=int, nargs='+', default=[21],
                        help='List of seeds for generation')
    parser.add_argument('--model_path', type=str, default='runwayml/stable-diffusion-v1-5',
                        help='Path to model or model identifier')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run the model on')

    # General parameters
    parser.add_argument('--guidance_scale', type=float, default=7.5,
                        help='Guidance scale for generation')
    parser.add_argument('--num_inference_steps', type=int, default=50,
                        help='Number of inference steps')

    # Pipeline-specific parameters
    parser.add_argument('--strength', type=float, default=0.8,
                        help='Strength for img2img pipeline')
    parser.add_argument('--edit_threshold', type=float, default=0.75,
                        help='Edit threshold for LEDITS')

    # DDPM-specific parameters
    parser.add_argument('--cfg_src', type=float, default=3.5,
                        help='Source classifier guidance scale for DDPM')
    parser.add_argument('--cfg_tar', type=float, default=15.0,
                        help='Target classifier guidance scale for DDPM')
    parser.add_argument('--skip', type=int, default=36,
                        help='Skip steps for DDPM')
    parser.add_argument('--xa', type=float, default=0.6,
                        help='Cross attention control for DDPM')
    parser.add_argument('--sa', type=float, default=0.2,
                        help='Self attention control for DDPM')
    parser.add_argument('--ddpm_mode', type=str, default='our_inv',
                        choices=['our_inv', 'p2pinv', 'p2pddim', 'ddim'],
                        help='DDPM inversion mode')

    return parser.parse_args()


def load_json_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return {item['image_name']: item['original_prompt'] for item in data}


def setup_ledits_pipeline(model_path, device):
    pipe = StableDiffusionPipeline_LEDITS.from_pretrained(model_path, safety_checker=None)
    pipe.scheduler = DPMSolverMultistepSchedulerInject.from_pretrained(
        model_path,
        subfolder="scheduler",
        algorithm_type="sde-dpmsolver++",
        solver_order=2
    )
    pipe.to(device)
    return pipe


def setup_img2img_pipeline(model_path, device):
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
        safety_checker=None
    )
    pipe.to(device)
    return pipe


def setup_ddpm_pipeline(model_path, device, args):
    if args.ddpm_mode in ["p2pddim", "ddim"]:
        scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False
        )
    else:
        scheduler = DDIMScheduler.from_config(model_path, subfolder="scheduler")

    pipe = StableDiffusionPipeline.from_pretrained(model_path).to(device)
    pipe.scheduler = scheduler
    pipe.scheduler.set_timesteps(args.num_inference_steps)
    return pipe


def process_with_ledits(pipe, image_path, prompt, output_path, seed, device, guidance_scale, edit_threshold):
    try:
        image = PIL.Image.open(image_path).convert("RGB")
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)

        _ = pipe.invert(image=image, num_inversion_steps=50, skip=0.1)
        edited_image = pipe(
            editing_prompt=[prompt],
            edit_guidance_scale=guidance_scale,
            edit_threshold=edit_threshold,
            generator=gen
        ).images[0]

        edited_image.save(output_path)
        return True
    except Exception as e:
        print(f"Error processing {image_path} with LEDITS (seed {seed}): {str(e)}")
        return False


def process_with_img2img(pipe, image_path, prompt, output_path, seed, device, guidance_scale, strength, num_inference_steps):
    try:
        image = PIL.Image.open(image_path).convert("RGB")
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)

        result = pipe(
            prompt=prompt,
            image=image,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=gen
        )

        result.images[0].save(output_path)
        return True
    except Exception as e:
        print(f"Error processing {image_path} with img2img (seed {seed}): {str(e)}")
        return False


def process_with_ddpm(pipe, image_path, prompt, output_path, seed, args):
    try:
        # Load and prepare image
        image = PIL.Image.open(image_path).convert("RGB")
        image = image.resize((512, 512))  # Ensure correct size
        image = torch.from_numpy(np.array(image)).float() / 127.5 - 1
        image = image.permute(2, 0, 1).unsqueeze(0).to(args.device)

        with autocast("cuda"), inference_mode():
            # Encode image
            latent = pipe.vae.encode(image).latent_dist.mode() * 0.18215

            # Forward process based on mode
            if args.ddpm_mode in ["p2pddim", "ddim"]:
                wT = pipe.scheduler.add_noise(
                    latent,
                    torch.randn_like(latent),
                    torch.tensor([args.num_inference_steps - 1])
                )
            else:
                # Standard forward process
                timesteps = pipe.scheduler.timesteps
                noise = torch.randn_like(latent)
                wT = pipe.scheduler.add_noise(latent, noise, timesteps[-1])

            # Set generator for deterministic results
            generator = torch.Generator(device=args.device)
            generator.manual_seed(seed)

            # Reverse process
            latents = pipe(
                prompt=prompt,
                latents=wT,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.cfg_tar,
                generator=generator
            ).images[0]

            latents.save(output_path)
        return True
    except Exception as e:
        print(f"Error processing {image_path} with DDPM (seed {seed}): {str(e)}")
        return False


def get_output_path(output_dir, image_name, seed, pipeline_type, ddpm_mode=""):
    base_name = os.path.splitext(image_name)[0]
    if pipeline_type == 'ddpm':
        new_name = f"{base_name}_{pipeline_type}_{ddpm_mode}_seed{seed}.png"
    else:
        new_name = f"{base_name}_{pipeline_type}_seed{seed}.png"
    return os.path.join(output_dir, new_name)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    prompt_data = load_json_data(args.json_path)

    # Setup appropriate pipeline
    if args.pipeline == 'ledits':
        pipe = setup_ledits_pipeline(args.model_path, args.device)
        process_func = process_with_ledits
    elif args.pipeline == 'img2img':
        pipe = setup_img2img_pipeline(args.model_path, args.device)
        process_func = process_with_img2img
    else:  # ddpm
        pipe = setup_ddpm_pipeline(args.model_path, args.device, args)
        process_func = process_with_ddpm

    successful = 0
    failed = 0
    total_iterations = len(prompt_data) * len(args.seeds)

    with tqdm(total=total_iterations, desc=f"Processing images with {args.pipeline}") as pbar:
        for image_name, prompt in prompt_data.items():
            input_path = os.path.join(args.input_dir, image_name)

            if not os.path.exists(input_path):
                print(f"Warning: Input image not found: {input_path}")
                failed += len(args.seeds)
                pbar.update(len(args.seeds))
                continue

            for seed in args.seeds:
                output_path = get_output_path(args.output_dir, image_name, seed, args.pipeline, args.ddpm_mode)

                if args.pipeline == 'ledits':
                    success = process_func(
                        pipe, input_path, prompt, output_path, seed, args.device,
                        args.guidance_scale, args.edit_threshold
                    )
                elif args.pipeline == 'img2img':
                    success = process_func(
                        pipe, input_path, prompt, output_path, seed, args.device,
                        args.guidance_scale, args.strength, args.num_inference_steps
                    )
                else:  # ddpm
                    success = process_func(
                        pipe, input_path, prompt, output_path, seed, args
                    )

                if success:
                    successful += 1
                else:
                    failed += 1

                pbar.update(1)

    # Print summary
    print(f"\nProcessing complete!")
    print(f"Pipeline used: {args.pipeline}")
    if args.pipeline == 'ddpm':
        print(f"DDPM mode: {args.ddpm_mode}")
    print(f"Successfully processed: {successful} images")
    print(f"Failed to process: {failed} images")
    print(f"Seeds used: {args.seeds}")
    print(f"Guidance scale: {args.guidance_scale}")
    if args.pipeline == 'img2img':
        print(f"Strength: {args.strength}")
    if args.pipeline == 'ddpm':
        print(f"Source CFG: {args.cfg_src}")
        print(f"Target CFG: {args.cfg_tar}")
        print(f"Skip steps: {args.skip}")
        print(f"XA: {args.xa}")
        print(f"SA: {args.sa}")
    print(f"Number of inference steps: {args.num_inference_steps}")


if __name__ == "__main__":
    main()