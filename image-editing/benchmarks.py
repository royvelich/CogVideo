import argparse
import json
import os
from pathlib import Path
import PIL
import torch
from tqdm import tqdm
import numpy as np
from torch import autocast, inference_mode
import calendar
import time
from diffusers import (
    StableDiffusionImg2ImgPipeline,
    StableDiffusionPipeline,
    DDIMScheduler
)
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from leditspp.scheduling_dpmsolver_multistep_inject import DPMSolverMultistepSchedulerInject
from leditspp import StableDiffusionPipeline_LEDITS

# Import DDPM-specific modules
from prompt_to_prompt.ptp_classes import AttentionStore, AttentionReplace, AttentionRefine, EmptyControl, load_512
from prompt_to_prompt.ptp_utils import register_attention_control, text2image_ldm_stable
from ddm_inversion.inversion_utils import inversion_forward_process, inversion_reverse_process
from ddm_inversion.ddim_inversion import ddim_inversion


class BlipCaptioner:
    def __init__(self, device):
        print("Initializing BLIP captioner...")
        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b"
        ).to(device)
        self.device = device

    def generate_caption(self, image):
        inputs = self.processor(image, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs)
        caption = self.processor.decode(outputs[0], skip_special_tokens=True)
        return caption


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
    parser.add_argument('--ddpm_mode', type=str, default='our_inv',
                        choices=['our_inv', 'p2pinv', 'p2pddim', 'ddim'],
                        help='DDPM inversion mode')
    parser.add_argument('--skip', type=int, default=36,
                        help='Skip steps for DDPM')
    parser.add_argument('--eta', type=float, default=1.0,
                        help='ETA parameter for DDPM')
    parser.add_argument('--xa', type=float, default=0.6,
                        help='Cross attention control for DDPM')
    parser.add_argument('--sa', type=float, default=0.2,
                        help='Self attention control for DDPM')

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


def process_with_ddpm(pipe, image_path, target_prompt, output_path, seed, args, blip_captioner):
    try:
        # Set random seed
        gen = torch.Generator(device=args.device)
        gen.manual_seed(seed)

        # Load and prepare image
        x0 = load_512(image_path, 0, 0, 0, 0, args.device)

        # Generate source prompt using BLIP
        source_prompt = blip_captioner.generate_caption(PIL.Image.open(image_path).convert("RGB"))
        # source_prompt = 'A sketch of a cat'
        print(f"BLIP generated source prompt: {source_prompt}")

        # Encode image with VAE
        with autocast("cuda"), inference_mode():
            w0 = (pipe.vae.encode(x0).latent_dist.mode() * 0.18215).float()

        # Forward process
        if args.ddpm_mode in ["p2pddim", "ddim"]:
            wT = ddim_inversion(pipe, w0, source_prompt, args.cfg_src)
        else:
            wt, zs, wts = inversion_forward_process(
                pipe, w0, etas=args.eta, prompt=source_prompt,
                cfg_scale=args.cfg_src, prog_bar=True,
                num_inference_steps=args.num_inference_steps
            )

        # Set up attention controller based on mode
        src_tar_len_eq = len(source_prompt.split()) == len(target_prompt.split())

        if args.ddpm_mode == "our_inv":
            controller = AttentionStore()
        elif args.ddpm_mode == "p2pinv":
            if src_tar_len_eq:
                controller = AttentionReplace(
                    [source_prompt, target_prompt],
                    args.num_inference_steps,
                    cross_replace_steps=args.xa,
                    self_replace_steps=args.sa,
                    model=pipe
                )
            else:
                controller = AttentionRefine(
                    [source_prompt, target_prompt],
                    args.num_inference_steps,
                    cross_replace_steps=args.xa,
                    self_replace_steps=args.sa,
                    model=pipe
                )
        elif args.ddpm_mode == "p2pddim":
            if src_tar_len_eq:
                controller = AttentionReplace(
                    [source_prompt, target_prompt],
                    args.num_inference_steps,
                    cross_replace_steps=0.8,
                    self_replace_steps=0.4,
                    model=pipe
                )
            else:
                controller = AttentionRefine(
                    [source_prompt, target_prompt],
                    args.num_inference_steps,
                    cross_replace_steps=0.8,
                    self_replace_steps=0.4,
                    model=pipe
                )
        else:  # ddim
            controller = EmptyControl()

        register_attention_control(pipe, controller)

        # Reverse process
        if args.ddpm_mode == "our_inv":
            w0, _ = inversion_reverse_process(
                pipe,
                xT=wts[args.skip],
                etas=args.eta,
                prompts=[target_prompt],
                cfg_scales=[args.cfg_tar],
                prog_bar=True,
                zs=zs[:args.skip],
                controller=controller
            )
        elif args.ddpm_mode == "p2pinv":
            w0, _ = inversion_reverse_process(
                pipe,
                xT=wts[args.num_inference_steps - args.skip],
                etas=args.eta,
                prompts=[source_prompt, target_prompt],
                cfg_scales=[args.cfg_src, args.cfg_tar],
                prog_bar=True,
                zs=zs[:(args.num_inference_steps - args.skip)],
                controller=controller
            )
            w0 = w0[1].unsqueeze(0)
        else:  # p2pddim or ddim
            if args.skip != 0:
                print("Skip parameter ignored for p2pddim/ddim mode")

            w0, _ = text2image_ldm_stable(
                pipe,
                [source_prompt, target_prompt],
                controller,
                args.num_inference_steps,
                [args.cfg_src, args.cfg_tar],
                None,
                wT
            )
            w0 = w0[1:2]

        # Decode final image
        with autocast("cuda"), inference_mode():
            x0_dec = pipe.vae.decode(1 / 0.18215 * w0).sample

        if x0_dec.dim() < 4:
            x0_dec = x0_dec.unsqueeze(0)

        # Save image
        image = PIL.Image.fromarray(
            (x0_dec[0].permute(1, 2, 0).cpu().numpy() * 127.5 + 128)
            .clip(0, 255)
            .astype(np.uint8)
        )
        image.save(output_path)

        # Save prompts
        prompt_info = {
            "source_prompt": source_prompt,
            "target_prompt": target_prompt
        }
        prompt_path = output_path.rsplit('.', 1)[0] + '_prompts.json'
        with open(prompt_path, 'w') as f:
            json.dump(prompt_info, f, indent=2)

        return True

    except Exception as e:
        print(f"Error processing {image_path} with DDPM (seed {seed}): {str(e)}")
        return False


def get_output_path(output_dir, image_name, seed, pipeline_type, ddpm_mode=""):
    base_name = os.path.splitext(image_name)[0]
    timestamp = calendar.timegm(time.gmtime())

    if pipeline_type == 'ddpm':
        new_name = f"{base_name}_{pipeline_type}_{ddpm_mode}_seed{seed}_{timestamp}.png"
    else:
        new_name = f"{base_name}_{pipeline_type}_seed{seed}_{timestamp}.png"
    return os.path.join(output_dir, new_name)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    prompt_data = load_json_data(args.json_path)

    # Initialize BLIP if using DDPM
    blip_captioner = None
    if args.pipeline == 'ddpm':
        blip_captioner = BlipCaptioner(args.device)

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
        for image_name, target_prompt in prompt_data.items():
            input_path = os.path.join(args.input_dir, image_name)

            if not os.path.exists(input_path):
                print(f"Warning: Input image not found: {input_path}")
                failed += len(args.seeds)
                pbar.update(len(args.seeds))
                continue

            for seed in args.seeds:
                output_path = get_output_path(args.output_dir, image_name, seed,
                                              args.pipeline, args.ddpm_mode)

                if args.pipeline == 'ddpm':
                    success = process_func(
                        pipe, input_path, target_prompt, output_path,
                        seed, args, blip_captioner
                    )
                elif args.pipeline == 'ledits':
                    success = process_func(
                        pipe, input_path, target_prompt, output_path,
                        seed, args.device, args.guidance_scale, args.edit_threshold
                    )
                else:  # img2img
                    success = process_func(
                        pipe, input_path, target_prompt, output_path,
                        seed, args.device, args.guidance_scale,
                        args.strength, args.num_inference_steps
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
        print(f"ETA: {args.eta}")
        print(f"XA: {args.xa}")
        print(f"SA: {args.sa}")
    print(f"Number of inference steps: {args.num_inference_steps}")


if __name__ == "__main__":
    main()