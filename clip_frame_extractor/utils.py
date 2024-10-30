import io
import os
import numpy as np
import torch
from torchvision import transforms

from decord import cpu, VideoReader, bridge
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import argparse
import gc

import cv2
import json
from typing import List, Union
import PIL

MODEL_PATH = "THUDM/cogvlm2-video-llama3-chat"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TORCH_TYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[
    0] >= 8 else torch.float16


tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

# Load the model
# if args.quant == 4:
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=TORCH_TYPE,
    trust_remote_code=True,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=TORCH_TYPE,
    ),
    low_cpu_mem_usage=True
).eval()


def analyze_frame_concise(
        frame  : Union[PIL.Image.Image, np.ndarray, torch.Tensor],
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        torch_type: torch.dtype = torch.float16
) -> str:
    """
    Analyzes a video frame and returns a single, concise sentence describing the main subject and its state.

    Args:
        frame: Video frame as PIL Image, numpy array, or tensor
        model: The vision-language model
        tokenizer: The tokenizer for the model
        device: Computing device
        torch_type: Torch dtype for computation

    Returns:
        str: A single sentence description of the frame
    """
    try:
        print("Input frame type:", type(frame))
        print("Frame size:", frame.size)
        print("Frame mode:", frame.mode)
        # Resize if needed (keeping aspect ratio)
        # max_size = 1024
        # if max(frame.size) > max_size:
        #     ratio = max_size / max(frame.size)
        #     new_size = tuple(int(dim * ratio) for dim in frame.size)
        #     frame = frame.resize(new_size, Image.Refilter.LANCZOS)
        #     print("Resized frame to:", frame.size)
        # image_size = 224
        # transform = transforms.Compose([
        #     transforms.PILToTensor(),  # Converts to tensor with values in [0, 255]
        #     transforms.Resize(image_size, antialias=True),  # Resize shorter side
        #
        # ])
        # frame_tensor = transform(frame)
        # # # Apply transform
        # # frame_tensor = transform(frame)  # Now frame is a 3D tensor (C, H, W)
        # #
        # # # Fix: Add batch dimension to make it 4D
        # frame_tensor = frame_tensor.unsqueeze(0)

        print("Building conversation inputs")
        inputs = model.build_conversation_input_ids(
            tokenizer=tokenizer,
            query="Describe the main subject and its state in this image in a single, clear sentence.",
            images=[frame_tensor],
            history=[],
            template_version='chat'
        )

        print("Preparing model inputs")
        inputs = {
            'input_ids': inputs['input_ids'].unsqueeze(0).to(device),
            'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to(device),
            'attention_mask': inputs['attention_mask'].unsqueeze(0).to(device),
            'images': [[inputs['images'][0].to(device).to(torch_type)]],
        }

        print("Generating description")
        with torch.no_grad():
            model.eval()
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                pad_token_id=128002,
                top_k=1,
                do_sample=True,
                top_p=0.1,
                temperature=0.3
            )
            print("Raw outputs shape:", outputs.shape)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            print("Sliced outputs shape:", outputs.shape)

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print("Raw response:", response)

            # Clean up the response
            description = ' '.join(response.strip().split())
            print("Final description:", description)
            return description

    except Exception as e:
        print(f"Error details: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Error analyzing frame: {str(e)}"

    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def generate_text_variations(input_text, device="cuda"):
    """
    Generate 10 variations of the input text with similar meaning.
    """
    strategy = 'chat'
    results = [input_text]  # Start with original text


    # List of 10 different query formats
    queries = [
        f"Complete this statement: {input_text}",
        f"Express this in your own words: {input_text}",
        f"Rephrase this observation: {input_text}",
        f"How would you describe this: {input_text}",
        f"write in another way: {input_text}",
        f"Provide an alternative description for: {input_text}",
        f"What's another way to express: {input_text}",
        f"Rewrite this statement: {input_text}",
        f"Share a different way to describe: {input_text}",
        f"Give me a variation of this phrase: {input_text}"
    ]


    try:
        for query in queries:

            inputs = model.build_conversation_input_ids(
                tokenizer=tokenizer,
                query=query,
                history=[],
                template_version=strategy
            )

            inputs = {
                'input_ids': inputs['input_ids'].unsqueeze(0).to(device),
                'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to(device),
                'attention_mask': inputs['attention_mask'].unsqueeze(0).to(device),
                'images': None,
            }

            gen_kwargs = {
                "max_new_tokens": 15,  # Short response is fine
                "pad_token_id": 128002,
                "do_sample": True,
                "top_k": 50,
                "top_p": 0.9,
                "temperature": 0.7,
                "num_beams": 3,
            }


            with torch.no_grad():
                model.eval()
                outputs = model.generate(**inputs, **gen_kwargs)

                outputs = outputs[:, inputs['input_ids'].shape[1]:]
                # print("Sliced outputs:", outputs)

                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                # print("Response:", response)

                # Clean up response from : and other special characters, remove text before :
                response = response.strip()
                response = response.split(":")[-1].strip()

                if response:
                    results.append(response)


    except Exception as e:
        print(f"Error generating variations: {str(e)}")

    finally:
        # Clean up
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()
        # gc.collect()
        pass

    return results



def detect_excessive_camera_movement(video_path, threshold=1.3):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # frame_step = max(1, total_frames // sample_frames)
    frame_step = 1

    prev_frame = None
    movement_scores = []

    for i in range(0, total_frames, frame_step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()

        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_frame is not None:
            # Calculate optical flow
            flow = cv2.calcOpticalFlowFarneback(prev_frame, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            # Calculate the magnitude of the flow vectors
            magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)

            # Calculate the average movement
            avg_movement = np.mean(magnitude)
            movement_scores.append(avg_movement)

        prev_frame = gray

    cap.release()

    if movement_scores:
        avg_movement_score = np.mean(movement_scores)
        max_movement_score = np.max(movement_scores)
        print(f"Average movement score: {avg_movement_score}")
        return avg_movement_score > threshold, avg_movement_score, max_movement_score
    else:
        print("No movement scores calculated")
        return False, np.inf, np.inf

def load_video(video_path):
    bridge.set_bridge('torch')
    with open(video_path, 'rb') as f:
        mp4_stream = f.read()
    num_frames = 8 # to avoid cuda out of memory

    decord_vr = VideoReader(io.BytesIO(mp4_stream), ctx=cpu(0))
    total_frames = len(decord_vr)
    timestamps = decord_vr.get_frame_timestamp(np.arange(total_frames))
    timestamps = [i[0] for i in timestamps]
    max_second = round(max(timestamps)) + 1
    frame_id_list = []
    for second in range(max_second):
        closest_num = min(timestamps, key=lambda x: abs(x - second))
        index = timestamps.index(closest_num)
        frame_id_list.append(index)
        if len(frame_id_list) >= num_frames:
            break
    video_data = decord_vr.get_batch(frame_id_list)
    video_data = video_data.permute(3, 0, 1, 2)
    print("video_data.shape: ", video_data.shape)
    return video_data


# elif args.quant == 8:
#     model = AutoModelForCausalLM.from_pretrained(
#         MODEL_PATH,
#         torch_dtype=TORCH_TYPE,
#         trust_remote_code=True,
#         quantization_config=BitsAndBytesConfig(
#             load_in_8bit=True,
#             bnb_4bit_compute_dtype=TORCH_TYPE,
#         ),
#         low_cpu_mem_usage=True
#     ).eval()
# else:
#     model = AutoModelForCausalLM.from_pretrained(
#         MODEL_PATH,
#         torch_dtype=TORCH_TYPE,
#         trust_remote_code=True
#     ).eval().to(DEVICE)

def generate_comparison_prompt(target_text):
    prompt = f"""First image:
Describe exactly what you see.

Then target text:
{target_text}

Looking at your image description,  what happens in the image that is not captured by the target text?  what is the difference between the two descriptions?"""

    return prompt

def generate_prompt_step_2(comparison_text, device="cuda"):
    prompt = f"""Regarding the differences mention here {comparison_text}, write in a clear and concise sentence whats happen in the image?"""
    try:

        results = []
        strategy = 'chat'
        inputs = model.build_conversation_input_ids(
            tokenizer=tokenizer,
            query=prompt,
            history=[],
            template_version=strategy
        )

        inputs = {
            'input_ids': inputs['input_ids'].unsqueeze(0).to(device),
            'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to(device),
            'attention_mask': inputs['attention_mask'].unsqueeze(0).to(device),
            'images': None,
        }

        gen_kwargs = {
            "max_new_tokens": 25,  # Short response is fine
            "pad_token_id": 128002,
            "do_sample": False,
            "top_k": 1,
            "top_p": 0.1,
            "temperature": 0.1,
            "num_beams": 1,
        }

        with torch.no_grad():
            model.eval()
            outputs = model.generate(**inputs, **gen_kwargs)

            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            # print("Sliced outputs:", outputs)

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # print("Response:", response)

            # Clean up response from : and other special characters, remove text before :
            response = response.strip()
            response = response.split(":")[-1].strip()

            if response:
                results.append(response)


    except Exception as e:
        print(f"Error generating variations: {str(e)}")

    finally:
        # Clean up
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()
        # gc.collect()
        pass

    return results


def process_video_to_text(video_path, last_frame_to_process=None, queries=None, max_new_tokens=2048):
    strategy = 'chat'
    video = load_video(video_path)
    print("video length before slicing: ", len(video))
    if last_frame_to_process is not None:
        video = video[:, :last_frame_to_process]
    if queries is None:
        queries = [
            "Describe the person's body position during the most intense action in 3-10 words. Use terms like 'extending', 'bending', or 'rotating'.",
            "How are the limbs arranged at the peak of the action? Answer in 3-10 words, considering words like 'thrust', 'swing', or 'reach'.",
            "Detail the body's pose at the most actionable moment in 3-10 words. Include terms such as 'arching', 'twisting', or 'lunging'.",
            "Explain the body's alignment during the crucial moment in 3-10 words. Consider 'leaning', 'balancing', or 'stretching'.",
            "What's the position of arms and legs at the key action's peak? Describe in 3-10 words, using terms like 'flexed', 'extended', or 'raised'.",
            "Describe the person's stance at the exact moment of main action in 3-10 words. Use words like 'crouching', 'pivoting', or 'standing'.",
            "How is the body configured during the most intense part of the movement? Use 3-10 words, with terms like 'contracting', 'expanding', or 'shifting'.",
            "Detail the active pose, focusing on limb positions during the key action in 3-10 words. Include 'gripping', 'pushing', or 'pulling'.",
            "Explain how the body is positioned at the climax of the movement in 3-10 words. Consider 'bracing', 'releasing', or 'impacting'.",
            "Describe the arrangement of body parts during the critical phase in 3-10 words. Use terms like 'aligned', 'angled', or 'positioned'."
        ]
    results = []
    model.eval()
    count = 0
    for query in queries:

        print("count: ", count)
        count += 1

        torch.cuda.empty_cache()

        inputs = model.build_conversation_input_ids(
            tokenizer=tokenizer,
            query=query,
            images=[video],
            history=[],
            template_version=strategy
        )

        inputs = {
            'input_ids': inputs['input_ids'].unsqueeze(0).to(DEVICE),
            'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to(DEVICE),
            'attention_mask': inputs['attention_mask'].unsqueeze(0).to(DEVICE),
            'images': [[inputs['images'][0].to(DEVICE).to(TORCH_TYPE)]],
        }
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": 128002,
            "top_k": 1,
            "do_sample": False, # maybe need True
            "top_p": 0.1,
            "temperature": 0.1,
        }
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            results.append(response)
        # clean cuda each iteration
        torch.cuda.empty_cache()
    gc.collect()

    return results





def extract_n_images_from_video(video_path: str, output_dir: str, n: int) -> List[str]:
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load the video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return []

    # Get the total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the step size
    step = max(1, total_frames // n)
    
    frame_paths = []
    for i in range(n):
        frame_number = min(i * step, total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()

        if ret:
            output_path = os.path.join(output_dir, f"frame_{i:04d}.jpg")
            cv2.imwrite(output_path, frame)
            frame_paths.append(output_path)
        else:
            print(f"Warning: Could not read frame {frame_number}")

    # Release the video capture object
    cap.release()

    print(f"{len(frame_paths)} frames extracted successfully.")
    return frame_paths



def main():
    video_dir = args.video_dir
    video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mov'))]

    for video_file in video_files:
        video_path = os.path.join(video_dir, video_file)
        print(f"\nProcessing video: {video_file}")

        try:
            if detect_excessive_camera_movement(video_path):
                print("Excessive camera movement detected. Skipping video.")
                continue
            instruction = process_video_to_text(video_path)
            # print("Description:", description)
            print("Instruction: ", instruction)
            # save the instruction to a file
        except Exception as e:
            print(f"Error processing {video_file}: {str(e)}")


def process_video_directory(video_dir: str, output_base_dir: str, n: int, json_output: str):
    video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mov'))]

    dataset = []

    for video_file in video_files:
        video_path = os.path.join(video_dir, video_file)
        print(f"\nProcessing video: {video_file}")

        try:
            instruction = process_video_to_text(video_path)
            # print("Description:", description)
            print("Instruction:", instruction)
            # save the instruction to a file
        except Exception as e:
            print(f"Error processing {video_file}: {str(e)}")


        video_name = os.path.splitext(video_file)[0]
        output_dir = os.path.join(output_base_dir, video_name)

        # Extract frames
        frame_paths = extract_n_images_from_video(video_path, output_dir, n)


        # Add to dataset
        dataset.append({
            "video_name": video_name,
            "frame_paths": frame_paths,
            "instruction": instruction
        })

    # Save dataset to JSON file
    with open(json_output, 'w') as f:
        json.dump(dataset, f, indent=2)

    print(f"Dataset saved to {json_output}")





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CogVLM2-Video Batch Processing")
    parser.add_argument('--quant', type=int, choices=[4, 8], help='Enable 4-bit or 8-bit precision loading', default=0)
    parser.add_argument('--video_dir', type=str, required=True, help='Directory containing video files')
    args = parser.parse_args()

    if 'int4' in MODEL_PATH:
        args.quant = 4
    main(args)

    video_directory = args.video_dir
    output_base_directory = "processed_dataset"
    number_of_frames = 8  # Change this to the desired number of frames per video
    json_output_file = "processed_dataset/dataset.json"

    process_video_directory(video_directory, output_base_directory, number_of_frames, json_output_file)