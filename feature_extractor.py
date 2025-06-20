# feature_extractor.py (Updated to load from two separate annotation files)

import os
import shutil
import json
import cv2
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tqdm import tqdm
import argparse
import config

# --- GPU Configuration (Optional but Recommended) ---
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


def create_vgg16_model():
    """Loads a pre-trained VGG16 model, modified for feature extraction."""
    base_model = VGG16(weights='imagenet', include_top=True)
    model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)
    return model


def extract_video_features(video_path, model):
    """Extracts VGG16 features from a video file."""
    if not os.path.exists(video_path):
        return None
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return None
    frame_indices = np.linspace(0, total_frames - 1, config.VIDEO_FRAMES, dtype=int)
    frames = []
    for i in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            img = cv2.resize(frame, config.VIDEO_IMG_SIZE)
            img = preprocess_input(img)
            frames.append(img)
    cap.release()
    if not frames:
        return None
    frames = np.array(frames)
    features = model.predict(frames, verbose=0)
    return features


def extract_audio_features(video_path):
    """Extracts MFCC features from the audio track of a video file."""
    if not os.path.exists(video_path):
        return None
    try:
        y, sr = librosa.load(video_path, sr=config.AUDIO_SAMPLING_RATE)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=config.AUDIO_N_MFCC)
        mfccs = mfccs.T
        if mfccs.shape[0] < config.AUDIO_TIME_STEPS:
            pad_width = config.AUDIO_TIME_STEPS - mfccs.shape[0]
            mfccs = np.pad(mfccs, ((0, pad_width), (0, 0)), mode='constant')
        else:
            mfccs = mfccs[:config.AUDIO_TIME_STEPS, :]
        return mfccs
    except Exception as e:
        return np.zeros((config.AUDIO_TIME_STEPS, config.AUDIO_N_MFCC))


def main():
    """Main function to run the feature extraction process."""
    parser = argparse.ArgumentParser(description='Extract video and audio features from the MSR-VTT dataset.')
    parser.add_argument('--mode', type=str, choices=['debug', 'full'], default='full',
                        help='Run in "debug" mode for a small subset or "full" mode for the entire dataset.')
    args = parser.parse_args()

    print(f"--- Starting Feature Extraction in {args.mode.upper()} MODE ---")

    os.makedirs(config.VIDEO_FEATURES_DIR, exist_ok=True)
    os.makedirs(config.AUDIO_FEATURES_DIR, exist_ok=True)

    print("Loading VGG16 model...")
    video_model = create_vgg16_model()
    print("VGG16 model loaded.")

    # --- MODIFIED: Load from two separate JSON files ---
    try:
        with open(config.TRAIN_VAL_ANNOTATION_FILE, 'r') as f:
            train_val_data = json.load(f)
        
        test_data = {'videos': []} # Default to an empty list
        if os.path.exists(config.TEST_ANNOTATION_FILE):
             with open(config.TEST_ANNOTATION_FILE, 'r') as f:
                test_data = json.load(f)
        else:
            print(f"Warning: Test annotation file not found at '{config.TEST_ANNOTATION_FILE}'. Skipping test videos.")

    except FileNotFoundError as e:
        print(f"--- ERROR: Annotation file not found! ---")
        print(e)
        print("Please make sure 'train_val_videodatainfo.json' is in the annotations folder.")
        exit()

    # Combine the video lists from both files, ensuring keys exist
    train_val_videos = train_val_data.get('videos', [])
    test_videos = test_data.get('videos', [])
    all_videos_info = train_val_videos + test_videos
    
    if not all_videos_info:
        print("--- ERROR: No video information found in the annotation files! Check your JSON files. ---")
        exit()
    # --- END MODIFICATION ---

    videos_to_process = []
    if args.mode == 'debug':
        print(f"DEBUG MODE: Processing {config.DEBUG_TRAIN_SAMPLES} train, "
              f"{config.DEBUG_VALID_SAMPLES} valid, and {config.DEBUG_TEST_SAMPLES} test samples.")
        
        train_samples = [v for v in all_videos_info if v['split'] == 'train'][:config.DEBUG_TRAIN_SAMPLES]
        valid_samples = [v for v in all_videos_info if v['split'] == 'validate'][:config.DEBUG_VALID_SAMPLES]
        test_samples = [v for v in all_videos_info if v['split'] == 'test'][:config.DEBUG_TEST_SAMPLES]
            
        videos_to_process = train_samples + valid_samples + test_samples
    else:
        videos_to_process = all_videos_info

    for video_info in tqdm(videos_to_process, desc="Extracting Features"):
        video_id = video_info['video_id']
        video_split = video_info['split']
        subfolder = 'train-val' if video_split in ['train', 'validate'] else 'test'
        video_file_name = f"{video_id}.mp4"
        video_path = os.path.join(config.VIDEO_DIR, subfolder, video_file_name)

        video_feature_path = os.path.join(config.VIDEO_FEATURES_DIR, f"{video_id}.npy")
        audio_feature_path = os.path.join(config.AUDIO_FEATURES_DIR, f"{video_id}.npy")

        if os.path.exists(video_feature_path) and os.path.exists(audio_feature_path):
            continue
        if not os.path.exists(video_path):
            # This is not an error, the dataset has some missing videos
            continue

        video_features = extract_video_features(video_path, video_model)
        if video_features is not None:
            np.save(video_feature_path, video_features)

        audio_features = extract_audio_features(video_path)
        if audio_features is not None:
            np.save(audio_feature_path, audio_features)

    print(f"\n--- Feature Extraction Complete ({args.mode.upper()} MODE) ---")
    print(f"Video features saved in: {config.VIDEO_FEATURES_DIR}")
    print(f"Audio features saved in: {config.AUDIO_FEATURES_DIR}")

if __name__ == '__main__':
    main()