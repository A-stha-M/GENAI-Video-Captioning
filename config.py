# config.py

import os

# --- PATHS ---
# Main project directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# MSRVTT dataset paths
DATA_DIR = os.path.join(BASE_DIR, 'data', 'MSRVTT')
VIDEO_DIR = os.path.join(DATA_DIR, 'videos')

# --- Updated: Paths for separate annotation files ---
ANNOTATIONS_DIR = os.path.join(DATA_DIR, 'annotations')
TRAIN_VAL_ANNOTATION_FILE = os.path.join(ANNOTATIONS_DIR, 'train_val_videodatainfo.json')
TEST_ANNOTATION_FILE = os.path.join(ANNOTATIONS_DIR, 'test_videodatainfo.json')

# Paths for extracted features
FEATURES_DIR = os.path.join(BASE_DIR, 'features')
VIDEO_FEATURES_DIR = os.path.join(FEATURES_DIR, 'video_vgg16')
AUDIO_FEATURES_DIR = os.path.join(FEATURES_DIR, 'audio_mfcc')

# Path for saving trained models and tokenizer
SAVED_MODELS_DIR = os.path.join(BASE_DIR, 'saved_models')

# --- FEATURE EXTRACTION ---
# Video settings
VIDEO_FRAMES = 80
VIDEO_IMG_SIZE = (224, 224)

# Audio settings
AUDIO_SAMPLING_RATE = 16000
AUDIO_N_MFCC = 13
AUDIO_TIME_STEPS = 80

# --- MODEL HYPERPARAMETERS ---
# Caption preprocessing
MAX_CAPTION_LENGTH = 20
VOCAB_SIZE = 5000

# Training settings
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001
LATENT_DIM = 512

# --- MODEL ARCHITECTURE ---
# Dimensions of the input features
VIDEO_FEATURE_DIM = 4096
AUDIO_FEATURE_DIM = AUDIO_N_MFCC

# --- DEBUG MODE ---
# Number of samples to use in debug mode
DEBUG_TRAIN_SAMPLES = 100
DEBUG_VALID_SAMPLES = 50
DEBUG_TEST_SAMPLES = 10 # Re-enabled test samples