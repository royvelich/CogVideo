import argparse

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from torch.nn.functional import cosine_similarity
import os
import shutil
from torchvision import transforms
from utils import process_video_to_text, detect_excessive_camera_movement, generate_text_variations, analyze_frame_concise, generate_comparison_prompt, generate_prompt_step_2

from scipy.ndimage import gaussian_filter1d
import json


def parse_args():
    parser = argparse.ArgumentParser(description="Your script description")
    parser.add_argument('--sampling_type', type=str, default='random',
                        choices=['random', 'clip', 'clip_diff'],
                        help='Type of sampling to use')
    # Add more arguments as needed, for example:
    parser.add_argument('--sample_size', type=int, default=2,
                        help='Number of frames to sample')
    parser.add_argument('--generate_videos', type=bool, default=False)
    return parser.parse_args()


def gaussian_smooth(data, sigma=5.0):
    return gaussian_filter1d(data, sigma)


def setup_output_directory(output_dir):
    os.makedirs(output_dir, exist_ok=True)

def extract_frames(video_path):
    if not os.path.exists(video_path):
        print(f"Error: Video file does not exist: {video_path}")
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = 1
    frames = []
    for i in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))

    cap.release()

    if not frames:
        print("Error: No frames were extracted from the video.")
        return None
    print("number of frames: ", len(frames))
    return frames

def get_text_embedding(text, model, processor, device):
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        return model.get_text_features(**inputs)

def get_text_embedding_as_mean_of_texts(texts, model, processor, device):
    text_embedding = torch.zeros(1, 768).to(device)
    for text in texts:
        text_embedding += get_text_embedding(text, model, processor, device)
    text_embedding /= len(texts)
    return text_embedding

def augment_and_embed_frames(frames, model, processor, device, num_augs=5):
    all_frame_embeddings = []
    augment_trans = transforms.Compose([
        transforms.RandomPerspective(fill=1, p=1, distortion_scale=0.5),
        transforms.RandomResizedCrop(224, scale=(0.7, 0.9)),
    ])

    for count, frame in enumerate(frames):
        frame_augs = [frame]
        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)

        frame = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1)

        # if count % 20 == 0:
        #     cv2.imwrite(f'{output_dir}/augmented_frame{count}.jpg', cv2.cvtColor(np.array(augment_trans(frame)).transpose(1, 2, 0), cv2.COLOR_RGB2BGR))

        for _ in range(num_augs - 1):
            pil_aug_frame = np.array(cv2.cvtColor(np.array(augment_trans(frame)).transpose(1, 2, 0), cv2.COLOR_RGB2BGR), dtype=np.uint8)
            frame_augs.append(pil_aug_frame)



        inputs = processor(images=frame_augs, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            aug_embeddings = model.get_image_features(**inputs)

        avg_embedding = torch.mean(aug_embeddings, dim=0, keepdim=True)
        all_frame_embeddings.append(avg_embedding)

    return all_frame_embeddings

def calculate_similarities(frame_embeddings, text_embedding):
    similarities = []
    for frame_embedding in frame_embeddings:
        if frame_embedding.size(0) != 1:
            frame_embedding = frame_embedding.squeeze(0)
        similarity = cosine_similarity(frame_embedding, text_embedding, dim=1).item()
        similarities.append(similarity)
    return similarities

def create_output_video(video_path, similarities, text, output_dir, output_video_path=None):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # title_text = text.replace(" ", "_").replace('.', '_')
    video_output_path = f'{output_video_path}/{text}_output_video.mkv'
    out = cv2.VideoWriter(video_output_path, fourcc, fps, (frame_width * 2, frame_height))

    max_similarity = max(similarities)
    min_similarity = min(similarities)

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        plt.figure(figsize=(10, 5))
        plt.plot(similarities[:frame_count+1], color='blue')
        plt.scatter(frame_count, similarities[frame_count], color='red')
        plt.ylim(min_similarity, max_similarity)
        plt.xlim(0, len(similarities))
        plt.xlabel('Frame')
        plt.ylabel('Cosine Similarity')
        plt.title(f'Cosine Similarity to "{text}"')

        plt.savefig(f'{output_dir}/temp_plot.png')
        plt.close()

        plot_img = cv2.imread(f'{output_dir}/temp_plot.png')
        plot_img = cv2.resize(plot_img, (frame_width, frame_height))

        combined_frame = np.hstack((frame, plot_img))
        out.write(combined_frame)

        frame_count += 1

        if frame_count % 10 == 0:
            print(f"Processed {frame_count} frames")

    cap.release()
    out.release()
    os.remove(f'{output_dir}/temp_plot.png')
    print(f"Processing complete. Output saved as: {video_output_path}")


def process_video_by_clip_sampling(video_path, texts, output_dir, model, preprocessor, device, optical_flow_stats : dict, video_num=0):
    frames = extract_frames(video_path)
    if frames is None:
        raise Exception("Error: No frames were extracted from the video.")

    text_embedding = get_text_embedding_as_mean_of_texts(texts, model=model, processor=preprocessor, device=device)
    frame_embeddings = augment_and_embed_frames(frames, model=model, processor=preprocessor, device=device)
    similarities = calculate_similarities(frame_embeddings, text_embedding)
    similarities = gaussian_smooth(similarities)

    save_source_target_image_text_dir = output_dir + '/' + texts[0] +'_video_num_' + str(video_num)
    save_source_target_image_text_dir = save_source_target_image_text_dir.replace(" ", "_").replace('.', '_')
    os.makedirs(save_source_target_image_text_dir, exist_ok=True)


    create_output_video(video_path, similarities, texts[0], output_dir=output_dir, output_video_path=save_source_target_image_text_dir)

    # save in the output directory the first and last frames
    # take the frame with the minimum and maximum cosine similarity and save them
    # min_cosine_similarity_index = similarities.index(min(similarities))
    # max_cosine_similarity_index = similarities.index(max(similarities))
    min_cosine_similarity_index = np.argmin(similarities)
    max_cosine_similarity_index = np.argmax(similarities)
    cv2.imwrite(f'{save_source_target_image_text_dir}/{texts[0]}min_cosine_similarity_frame.jpg', cv2.cvtColor(np.array(frames[min_cosine_similarity_index]), cv2.COLOR_RGB2BGR))
    cv2.imwrite(f'{save_source_target_image_text_dir}/{texts[0]}max_cosine_similarity_frame.jpg', cv2.cvtColor(np.array(frames[max_cosine_similarity_index]), cv2.COLOR_RGB2BGR))
    # save text[0]
    # with open(f'{save_source_target_image_text_dir}/text.txt', 'w') as f:
    #     f.write(texts[0])
    # with open(f'{save_source_target_image_text_dir}/optical_flow_stats.txt', 'w') as f:
    #     f.write(str(optical_flow_stats))
    # save text and optical flow stats as json
    with open(f'{save_source_target_image_text_dir}/text.json', 'w') as f:
        json.dump(texts[0], f)
    with open(f'{save_source_target_image_text_dir}/optical_flow_stats.json', 'w') as f:
        json.dump(optical_flow_stats, f)

def process_video_by_clip_diff_sampling(video_path, target_texts, output_dir, model, preprocessor, device, optical_flow_stats : dict, video_num=0):
    frames = extract_frames(video_path)
    if frames is None:
        raise Exception("Error: No frames were extracted from the video.")



    # source_text = 'An open book' # for debug
    # target_text = 'A closed book' # for debug
    # source_text = 'A man standing'
    # target_text = 'A man giving a thumbs up'
    # source_text_embedding = get_text_embedding(source_text, model=model, processor=preprocessor, device=device)
    # target_text_embedding = get_text_embedding(target_text, model=model, processor=preprocessor, device=device)
    # source_text = analyze_frame_concise(frames[0], device)
    # source_query = generate_comparison_prompt(target_texts[0])
    source_text = process_video_to_text(video_path, last_frame_to_process=1, queries=[ "Describe the main subject and its state in this image in a concise, single, clear sentence."])
    # source_text = generate_prompt_step_2(source_text)

    print("source_text: ", source_text)
    source_text_augmented = generate_text_variations(source_text[0], device)
    source_text_embedding = get_text_embedding_as_mean_of_texts(source_text_augmented, model=model, processor=preprocessor, device=device)
    target_text_embedding = get_text_embedding_as_mean_of_texts(target_texts, model=model, processor=preprocessor, device=device)

    text_embeddings_diff = target_text_embedding - source_text_embedding
    frame_embeddings = augment_and_embed_frames(frames, model=model, processor=preprocessor, device=device)
    frame_embeddings_diff_from_first_frame = []
    for frame_embedding in frame_embeddings:
        frame_embeddings_diff_from_first_frame.append(frame_embedding - frame_embeddings[0])

    similarities = calculate_similarities(frame_embeddings_diff_from_first_frame, text_embeddings_diff)
    similarities = gaussian_smooth(similarities)

    save_source_target_image_text_dir = output_dir + '/' + target_texts[0] + '_video_num_' + str(video_num)
    save_source_target_image_text_dir = save_source_target_image_text_dir.replace(" ", "_").replace('.', '_')
    os.makedirs(save_source_target_image_text_dir, exist_ok=True)

    create_output_video(video_path, similarities, target_texts[0], output_dir=output_dir,
                        output_video_path=save_source_target_image_text_dir)

    # save in the output directory the first and last frames
    # take the frame with the minimum and maximum cosine similarity and save them
    # min_cosine_similarity_index = similarities.index(min(similarities))
    # max_cosine_similarity_index = similarities.index(max(similarities))
    min_cosine_similarity_index = np.argmin(similarities)
    max_cosine_similarity_index = np.argmax(similarities)
    cv2.imwrite(f'{save_source_target_image_text_dir}/{target_texts[0]}min_cosine_similarity_frame.jpg',
                cv2.cvtColor(np.array(frames[min_cosine_similarity_index]), cv2.COLOR_RGB2BGR))
    cv2.imwrite(f'{save_source_target_image_text_dir}/{target_texts[0]}max_cosine_similarity_frame.jpg',
                cv2.cvtColor(np.array(frames[max_cosine_similarity_index]), cv2.COLOR_RGB2BGR))
    # After calculating initial similarities
    num_frames = len(similarities)
    max_sim = np.max(similarities)

    def reg_loss(t):
        # Exponential increase towards end
        return np.exp( 1.2*t / num_frames)  # Much steeper curve

    # Convert similarities to losses (lower is better)

    # Apply the regularization formula: loss_sim * reg_loss^sign(loss_sim)
    similarities_regularized = np.array([similarities[t] * (reg_loss(t) ** np.sign(similarities[t])) for t in range(num_frames)])
    similarities_regularized_max_index = np.argmax(similarities_regularized)
    cv2.imwrite(f'{save_source_target_image_text_dir}/{target_texts[0]}similarities_regularized_max_index.jpg',
                cv2.cvtColor(np.array(frames[similarities_regularized_max_index]), cv2.COLOR_RGB2BGR))

    print("max cosine similarity index: ", max_cosine_similarity_index)
    print("final losses min index: ", similarities_regularized_max_index)

    # save text[0]
    # with open(f'{save_source_target_image_text_dir}/text.txt', 'w') as f:
    #     f.write(texts[0])
    # with open(f'{save_source_target_image_text_dir}/optical_flow_stats.txt', 'w') as f:
    #     f.write(str(optical_flow_stats))
    # save text and optical flow stats as json
    with open(f'{save_source_target_image_text_dir}/text.json', 'w') as f:
        json.dump(target_texts[0], f)
    with open(f'{save_source_target_image_text_dir}/optical_flow_stats.json', 'w') as f:
        json.dump(optical_flow_stats, f)


def process_video_by_random_sampling(video_path, output_dir, texts, sample_size=2, video_num=0, optical_flow_stats : dict = None):
    frames = extract_frames(video_path)
    if frames is None:
        raise Exception("Error: No frames were extracted from the video.")
    total_frames = len(frames)
    # choose the first frame randomly between the first total_frames//sample_size frames and then sample the rest each in the next total_frames//sample_size
    first_frame_index = np.random.randint(0, total_frames//sample_size)
    sampled_frames = [frames[first_frame_index]]
    print("first frame index: ", first_frame_index)
    print("total frames: ", total_frames)

    for i in range(1, sample_size):
        print(" i*total_frames//sample_size", i*total_frames//sample_size)
        print(" (i+1)*total_frames//sample_size", (i+1)*total_frames//sample_size)
        sampled_frames.append(frames[np.random.randint(i*total_frames//sample_size, (i+1)*total_frames//sample_size)])
    # save the sampled frames
    save_source_target_image_text_dir = output_dir + '/' + 'random_sampling' + '_video_num_' + str(video_num)
    save_source_target_image_text_dir = save_source_target_image_text_dir.replace(" ", "_").replace('.', '_')
    os.makedirs(save_source_target_image_text_dir, exist_ok=True)
    for i, frame in enumerate(sampled_frames):
        cv2.imwrite(f'{save_source_target_image_text_dir}/frame_{i}.jpg', cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))


    print(f"Processed video: {video_path}")

    # save text and optical flow stats as json
    with open(f'{save_source_target_image_text_dir}/text.json', 'w') as f:
        json.dump(texts[0], f)
    with open(f'{save_source_target_image_text_dir}/optical_flow_stats.json', 'w') as f:
        json.dump(optical_flow_stats, f)





def augment_text(text,  device=None):
    # send the text to cog and get 10 augmented texts with the same meaning


    texts = generate_text_variations(text,device)
    print("augmented texts: ")
    print(texts)

    return texts


def process_directory(directory_path, output_dir, sampling_type, sample_size=2, model = None, processor = None, device = None):
    video_num = 0
    for filename in os.listdir(directory_path):
        # if there are directories
        if filename.endswith((".mp4", ".avi", ".mov")):
            video_path = os.path.join(directory_path, filename)
            # if avg optical flow is high, then it can be considered as excessive camera movement.
            # if max optical flow is high, then it can be considered as scene change - may be false, need to check.
            is_excessive_movement, avg_optical_flow, max_optical_flow = detect_excessive_camera_movement(video_path)
            if is_excessive_movement:
                print("video_path: ", video_path)
                print("Excessive camera movement detected. Skipping video.")
                # continue
            # texts = process_video_to_text(video_path=video_path)
            # texts = ["The chair gradually splits in half, revealing clean saw marks as the two halves slowly separate."]
            # file_name_without_extension = os.path.splitext(filename)[0]
            # texts = [file_name_without_extension]
            # load annotations.json file
            with open('annotations.json', 'r') as f:
                annotations = json.load(f)

            print(filename)
            # take from annotations the first n letters that represent a number names of the file as a key - need to check when the first '_'
            end_num_pos = filename.find('_')
            if end_num_pos == -1:
                end_num_pos = filename.find('.')
            entry = annotations[int(filename[:end_num_pos])-1]

            # entry = annotations[int(filename[:2])-1]
            texts = [entry["original_prompt"]]
            print("texts: ", texts[0])
            texts = augment_text(texts[0])



            optical_flow_stats = {"avg_optical_flow": float(avg_optical_flow), "max_optical_flow": float(max_optical_flow), "is_excessive_movement": bool(is_excessive_movement)}
            print(f"Processing video: {filename}")
            if sampling_type == 'clip':
                process_video_by_clip_sampling(video_path, texts, video_num=video_num, output_dir=output_dir, model=model, preprocessor=processor, device=device, optical_flow_stats=optical_flow_stats)
            elif sampling_type == 'clip_diff':
                process_video_by_clip_diff_sampling(video_path, target_texts=texts, video_num=video_num, output_dir=output_dir, model=model, preprocessor=processor, device=device, optical_flow_stats=optical_flow_stats)

            else:
                process_video_by_random_sampling(video_path, output_dir, texts=texts, sample_size=sample_size, video_num=video_num, optical_flow_stats=optical_flow_stats)
            video_num += 1

def main():
    args = parse_args()

    print(f"Sampling type: {args.sampling_type}")
    print(f"Sample size: {args.sample_size}")

    directory_path = "/home/gal.yona/ImageEditingThroughVideos/ImageVideoEditing/clip_frame_extractor/videos_to_process"

    if args.sampling_type == 'clip' or args.sampling_type == 'clip_diff':
        print("Using CLIP model for sampling")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        output_dir = "cosine_similarity_generated_videos_output"

        setup_output_directory(output_dir=output_dir)

        # Hardcoded text prompts
        # texts = [
        #     "Extend your arm to strike the punching bag.",
        #     "Thrust your fist forward towards the bag.",
        #     "Propel your hand to hit the hanging bag.",
        #     "Launch your arm to make contact with the bag.",
        #     "Project your fist to punch the training bag.",
        #     "Drive your hand forward to impact the bag.",
        #     "Stretch out your arm to connect with the bag.",
        #     "Reach out swiftly to strike the punching bag.",
        #     "Shoot your fist straight at the hanging bag.",
        #     "Lengthen your arm to deliver a punch to the bag."
        # ]

        # Process a single video
        # video_path = "/path/to/your/video.mp4"
        # process_video(video_path, texts)

        # Process a directory of videos
        # directory_path = "/mnt/gipnetapp_public/moments_in_time/Moments_in_Time_Raw/training"
        process_directory(directory_path, output_dir=output_dir, sampling_type=args.sampling_type, model=model, processor=processor, device=device)

    elif args.sampling_type == 'random':
        output_dir = "random_sampling_output"

        setup_output_directory(output_dir=output_dir)
        print("Using random sampling for frames")

        process_directory(directory_path, output_dir, sampling_type=args.sampling_type, sample_size=args.sample_size)
    else:
        print("Invalid sampling type. Please choose 'random', 'clip' or 'clip_diff.")



if __name__ == "__main__":
    main()