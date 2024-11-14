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
    DDIMScheduler,
    DDIMInverseScheduler,
    StableDiffusionPix2PixZeroPipeline
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
    parser = argparse.ArgumentParser(description='Process images using multiple pipelines')
    # General arguments
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing input images')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory for output images')
    parser.add_argument('--json_path', type=str, required=True, help='Path to JSON file with prompts')
    parser.add_argument('--pipeline', type=str, choices=['ledits', 'img2img', 'ddpm', 'pix2pix'], required=True,
                        help='Which pipeline to use')
    parser.add_argument('--seeds', type=int, nargs='+', default=[21],
                        help='List of seeds for generation')
    parser.add_argument('--model_path', type=str, default='runwayml/stable-diffusion-v1-5',
                        help='Path to model or model identifier')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run the model on')

    # LEDITS-specific parameters
    parser.add_argument('--ledits_guidance_scale', type=float, default=10.0,
                        help='Guidance scale for LEDITS generation')
    parser.add_argument('--ledits_num_inference_steps', type=int, default=50,
                        help='Number of inference steps for LEDITS')
    parser.add_argument('--ledits_edit_threshold', type=float, default=0.75,
                        help='Edit threshold for LEDITS')

    # img2img-specific parameters
    parser.add_argument('--img2img_guidance_scale', type=float, default=7.5,
                        help='Guidance scale for img2img generation')
    parser.add_argument('--img2img_num_inference_steps', type=int, default=50,
                        help='Number of inference steps for img2img')
    parser.add_argument('--img2img_strength', type=float, default=0.75,
                        help='Strength for img2img pipeline')

    # DDPM-specific parameters
    parser.add_argument('--ddpm_cfg_src', type=float, default=3.5,
                        help='Source classifier guidance scale for DDPM')
    parser.add_argument('--ddpm_cfg_tar', type=float, default=15.0,
                        help='Target classifier guidance scale for DDPM')
    parser.add_argument('--ddpm_mode', type=str, default='our_inv',
                        choices=['our_inv', 'p2pinv', 'p2pddim', 'ddim'],
                        help='DDPM inversion mode')
    parser.add_argument('--ddpm_num_inference_steps', type=int, default=50,
                        help='Number of inference steps for DDPM')
    parser.add_argument('--ddpm_skip', type=int, default=36,
                        help='Skip steps for DDPM')
    parser.add_argument('--ddpm_eta', type=float, default=1.0,
                        help='ETA parameter for DDPM')
    parser.add_argument('--ddpm_xa', type=float, default=0.6,
                        help='Cross attention control for DDPM')
    parser.add_argument('--ddpm_sa', type=float, default=0.2,
                        help='Self attention control for DDPM')

    # Pix2Pix-specific parameters
    parser.add_argument('--pix2pix_guidance_scale', type=float, default=7.5,
                        help='Guidance scale for Pix2Pix generation')
    parser.add_argument('--pix2pix_num_inference_steps', type=int, default=50,
                        help='Number of inference steps for Pix2Pix')
    parser.add_argument('--pix2pix_cross_attention_guidance_amount', type=float, default=0.15,
                        help='Cross attention guidance amount for Pix2Pix Zero')

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
    pipe.scheduler.set_timesteps(args.ddpm_num_inference_steps)
    return pipe


def setup_pix2pix_pipeline(model_path, device):
    pipe = StableDiffusionPix2PixZeroPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
        safety_checker=None
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.inverse_scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)
    return pipe


def generate_embeddings(pipe, prompts, batch_size=2):
    """Generate embeddings for a list of prompts"""
    return pipe.get_embeds(prompts, batch_size=batch_size)


def get_output_path(output_dir, image_name, seed, pipeline_type, ddpm_mode=""):
    # Create method-specific subfolder
    method_folder = pipeline_type
    if pipeline_type == 'ddpm':
        method_folder = f"{pipeline_type}_{ddpm_mode}"
    method_path = os.path.join(output_dir, method_folder)

    # Create seed-specific subfolder
    seed_path = os.path.join(method_path, f"seed_{seed}")

    # Create all necessary directories
    os.makedirs(seed_path, exist_ok=True)

    # Create filename
    base_name = os.path.splitext(image_name)[0]
    # timestamp = calendar.timegm(time.gmtime())
    # new_name = f"{base_name}_{timestamp}.png"
    new_name = f"{base_name}.png"

    return os.path.join(seed_path, new_name)


def process_with_ledits(pipe, image_path, prompt, output_path, seed, device, args):
    try:
        # Set random seed with CPU generator
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

        image = PIL.Image.open(image_path).convert("RGB")
        _ = pipe.invert(image, num_inversion_steps=50, skip=0.1)
        edited_image = pipe(
            editing_prompt=[prompt],
            edit_guidance_scale=args.ledits_guidance_scale,
            edit_threshold=args.ledits_edit_threshold,
        ).images[0]

        edited_image.save(output_path)
        return True
    except Exception as e:
        print(f"Error processing {image_path} with LEDITS (seed {seed}): {str(e)}")
        return False


def process_with_img2img(pipe, image_path, prompt, output_path, seed, device, args):
    try:
        # Set random seed with CPU generator
        generator = torch.Generator(device='cpu')
        generator.manual_seed(seed)

        image = PIL.Image.open(image_path).convert("RGB")
        result = pipe(
            prompt=prompt,
            image=image,
            strength=args.img2img_strength,
            guidance_scale=args.img2img_guidance_scale,
            num_inference_steps=args.img2img_num_inference_steps,
            generator=generator
        )

        result.images[0].save(output_path)
        return True
    except Exception as e:
        print(f"Error processing {image_path} with img2img (seed {seed}): {str(e)}")
        return False


def process_with_ddpm(pipe, image_path, target_prompt, output_path, seed, args, blip_captioner):
    try:
        # Set random seed with CPU generator
        generator = torch.Generator(device='cpu')
        generator.manual_seed(seed)

        # Load and prepare image
        x0 = load_512(image_path, 0, 0, 0, 0, args.device)

        # Generate source prompt using BLIP
        # source_prompt = blip_captioner.generate_caption(PIL.Image.open(image_path).convert("RGB"))
        source_prompt = 'A man standing naturally with his arms relaxed at his sides.'
        print(f"BLIP generated source prompt: {source_prompt}")

        # Encode image with VAE
        with autocast("cuda"), inference_mode():
            w0 = (pipe.vae.encode(x0).latent_dist.mode() * 0.18215).float()

        # Forward process
        if args.ddpm_mode in ["p2pddim", "ddim"]:
            wT = ddim_inversion(pipe, w0, source_prompt, args.ddpm_cfg_src)
        else:
            wt, zs, wts = inversion_forward_process(
                pipe, w0, etas=args.ddpm_eta, prompt=source_prompt,
                cfg_scale=args.ddpm_cfg_src, prog_bar=True,
                num_inference_steps=args.ddpm_num_inference_steps
            )

        # Set up attention controller based on mode
        src_tar_len_eq = len(source_prompt.split()) == len(target_prompt.split())

        if args.ddpm_mode == "our_inv":
            controller = AttentionStore()
        elif args.ddpm_mode == "p2pinv":
            if src_tar_len_eq:
                controller = AttentionReplace(
                    [source_prompt, target_prompt],
                    args.ddpm_num_inference_steps,
                    cross_replace_steps=args.ddpm_xa,
                    self_replace_steps=args.ddpm_sa,
                    model=pipe
                )
            else:
                controller = AttentionRefine(
                    [source_prompt, target_prompt],
                    args.ddpm_num_inference_steps,
                    cross_replace_steps=args.ddpm_xa,
                    self_replace_steps=args.ddpm_sa,
                    model=pipe
                )
        elif args.ddpm_mode == "p2pddim":
            if src_tar_len_eq:
                controller = AttentionReplace(
                    [source_prompt, target_prompt],
                    args.ddpm_num_inference_steps,
                    cross_replace_steps=0.8,
                    self_replace_steps=0.4,
                    model=pipe
                )
            else:
                controller = AttentionRefine(
                    [source_prompt, target_prompt],
                    args.ddpm_num_inference_steps,
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
                xT=wts[args.ddpm_skip],
                etas=args.ddpm_eta,
                prompts=[target_prompt],
                cfg_scales=[args.ddpm_cfg_tar],
                prog_bar=True,
                zs=zs[:args.ddpm_skip],
                controller=controller
            )
        elif args.ddpm_mode == "p2pinv":
            w0, _ = inversion_reverse_process(
                pipe,
                xT=wts[args.ddpm_num_inference_steps - args.ddpm_skip],
                etas=args.ddpm_eta,
                prompts=[source_prompt, target_prompt],
                cfg_scales=[args.ddpm_cfg_src, args.ddpm_cfg_tar],
                prog_bar=True,
                zs=zs[:(args.ddpm_num_inference_steps - args.ddpm_skip)],
                controller=controller
            )
            w0 = w0[1].unsqueeze(0)
        else:  # p2pddim or ddim
            if args.ddpm_skip != 0:
                print("Skip parameter ignored for p2pddim/ddim mode")

            w0, _ = text2image_ldm_stable(
                pipe,
                [source_prompt, target_prompt],
                controller,
                args.ddpm_num_inference_steps,
                [args.ddpm_cfg_src, args.ddpm_cfg_tar],
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


def process_with_pix2pix(pipe, image_path, target_prompt, output_path, seed, args, blip_captioner):
    try:
        # Set random seed with CPU generator
        generator = torch.Generator(device='cpu')
        generator.manual_seed(seed)

        # Load and prepare image
        image = PIL.Image.open(image_path).convert("RGB").resize((512, 512))

        # Generate source prompt using BLIP
        # source_prompt = blip_captioner.generate_caption(image)
        source_prompt = 'A man standing naturally with his arms relaxed at his sides.'
        print(f"BLIP generated source prompt: {source_prompt}")

        # Generate source and target embeddings
        source_prompts = [
            source_prompt,
        ]
        target_prompts = [
            target_prompt,
        ]

        # Get embeddings
        source_embeds = generate_embeddings(pipe, source_prompts)
        target_embeds = generate_embeddings(pipe, target_prompts)

        # Get inverted latents
        inv_latents = pipe.invert(
            source_prompt,
            image=image,
            generator=generator
        ).latents

        # Generate edited image
        result = pipe(
            source_prompt,
            source_embeds=source_embeds,
            target_embeds=target_embeds,
            num_inference_steps=args.pix2pix_num_inference_steps,
            cross_attention_guidance_amount=args.pix2pix_cross_attention_guidance_amount,
            guidance_scale=args.pix2pix_guidance_scale,
            generator=generator,
            latents=inv_latents,
            negative_prompt=source_prompt,
        )

        # Save image
        result.images[0].save(output_path)

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
        print(f"Error processing {image_path} with Pix2Pix Zero (seed {seed}): {str(e)}")
        return False


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    prompt_data = load_json_data(args.json_path)

    # Initialize BLIP if using DDPM or Pix2Pix
    blip_captioner = None
    if args.pipeline in ['ddpm', 'pix2pix']:
        blip_captioner = BlipCaptioner(args.device)

    # Setup appropriate pipeline
    if args.pipeline == 'ledits':
        pipe = setup_ledits_pipeline(args.model_path, args.device)
        process_func = process_with_ledits
    elif args.pipeline == 'img2img':
        pipe = setup_img2img_pipeline(args.model_path, args.device)
        process_func = process_with_img2img
    elif args.pipeline == 'pix2pix':
        pipe = setup_pix2pix_pipeline(args.model_path, args.device)
        process_func = process_with_pix2pix
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
                elif args.pipeline == 'pix2pix':
                    success = process_func(
                        pipe, input_path, target_prompt, output_path,
                        seed, args, blip_captioner
                    )
                elif args.pipeline == 'ledits':
                    success = process_func(
                        pipe, input_path, target_prompt, output_path,
                        seed, args.device, args
                    )
                else:  # img2img
                    success = process_func(
                        pipe, input_path, target_prompt, output_path,
                        seed, args.device, args
                    )

                if success:
                    successful += 1
                else:
                    failed += 1

                pbar.update(1)

    # Print summary
    print(f"\nProcessing complete!")
    print(f"Pipeline used: {args.pipeline}")
    print(f"Successfully processed: {successful} images")
    print(f"Failed to process: {failed} images")
    print(f"Seeds used: {args.seeds}")

    # Print method-specific parameters
    if args.pipeline == 'ledits':
        print(f"LEDITS parameters:")
        print(f"  Guidance scale: {args.ledits_guidance_scale}")
        print(f"  Edit threshold: {args.ledits_edit_threshold}")
        print(f"  Inference steps: {args.ledits_num_inference_steps}")

    elif args.pipeline == 'img2img':
        print(f"img2img parameters:")
        print(f"  Guidance scale: {args.img2img_guidance_scale}")
        print(f"  Strength: {args.img2img_strength}")
        print(f"  Inference steps: {args.img2img_num_inference_steps}")

    elif args.pipeline == 'ddpm':
        print(f"DDPM parameters:")
        print(f"  Mode: {args.ddpm_mode}")
        print(f"  Source CFG: {args.ddpm_cfg_src}")
        print(f"  Target CFG: {args.ddpm_cfg_tar}")
        print(f"  Skip steps: {args.ddpm_skip}")
        print(f"  ETA: {args.ddpm_eta}")
        print(f"  XA: {args.ddpm_xa}")
        print(f"  SA: {args.ddpm_sa}")
        print(f"  Inference steps: {args.ddpm_num_inference_steps}")

    elif args.pipeline == 'pix2pix':
        print(f"Pix2Pix parameters:")
        print(f"  Guidance scale: {args.pix2pix_guidance_scale}")
        print(f"  Cross attention guidance amount: {args.pix2pix_cross_attention_guidance_amount}")
        print(f"  Inference steps: {args.pix2pix_num_inference_steps}")


if __name__ == "__main__":
    main()