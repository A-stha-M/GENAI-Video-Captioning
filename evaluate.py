# evaluate.py

import os
import json
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import tensorflow as tf
from tqdm import tqdm
import pickle
import config

# --- GPU Configuration (Optional but Recommended) ---
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


def load_inference_models():
    """Loads the trained encoder and decoder models."""
    encoder_model_path = os.path.join(config.SAVED_MODELS_DIR, 'encoder_model.h5')
    decoder_model_path = os.path.join(config.SAVED_MODELS_DIR, 'decoder_model.h5')

    if not os.path.exists(encoder_model_path) or not os.path.exists(decoder_model_path):
        print("--- ERROR ---")
        print("Trained model files not found. Please train the model first by running 'train.py'.")
        exit()
        
    inf_encoder_model = load_model(encoder_model_path)
    inf_decoder_model = load_model(decoder_model_path)
    return inf_encoder_model, inf_decoder_model


def load_tokenizer():
    """Loads the saved tokenizer."""
    tokenizer_path = os.path.join(config.SAVED_MODELS_DIR, 'tokenizer.pkl')
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    return tokenizer


def greedy_search_prediction(video_features, audio_features, tokenizer, encoder_model, decoder_model):
    """
    Generates a caption for a video using greedy search.
    This version is updated to take both video and audio features.
    """
    index_to_word = {value: key for key, value in tokenizer.word_index.items()}
    
    # Reshape features to be (1, timesteps, feature_dim)
    video_features = np.expand_dims(video_features, axis=0)
    audio_features = np.expand_dims(audio_features, axis=0)

    # Encode the input features to get the initial state of the decoder
    states_value = encoder_model.predict([video_features, audio_features], verbose=0)

    # Start with the <bos> token
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = tokenizer.word_index['<bos>']
    
    decoded_sentence = ''
    
    for _ in range(config.MAX_CAPTION_LENGTH):
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value, verbose=0)
        
        # Get the most likely next word
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = index_to_word.get(sampled_token_index, None)

        if sampled_word is None or sampled_word == '<eos>':
            break
            
        decoded_sentence += ' ' + sampled_word
        
        # Update the target sequence for the next iteration
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        
        # Update states
        states_value = [h, c]
        
    return decoded_sentence.strip()


def main():
    """Main function to run evaluation on the test set."""
    print("--- Starting Evaluation on Test Set ---")

    # Load models and tokenizer
    print("Loading trained models and tokenizer...")
    encoder_model, decoder_model = load_inference_models()
    tokenizer = load_tokenizer()
    print("Models and tokenizer loaded.")

    # Get the list of test videos and their ground truth captions
    with open(config.TEST_ANNOTATION_FILE, 'r') as f:
        test_data = json.load(f)
    
    test_videos = {item['video_id']: [] for item in test_data['videos']}
    for sentence_item in test_data['sentences']:
        if sentence_item['video_id'] in test_videos:
            test_videos[sentence_item['video_id']].append(sentence_item['caption'])

    results = []
    
    # Loop through all test videos
    for video_id in tqdm(test_videos.keys(), desc="Generating Captions"):
        video_feature_path = os.path.join(config.VIDEO_FEATURES_DIR, f"{video_id}.npy")
        audio_feature_path = os.path.join(config.AUDIO_FEATURES_DIR, f"{video_id}.npy")

        # Check if feature files exist
        if not os.path.exists(video_feature_path) or not os.path.exists(audio_feature_path):
            # print(f"Warning: Feature files for {video_id} not found. Skipping.")
            continue
            
        # Load the features
        video_features = np.load(video_feature_path)
        audio_features = np.load(audio_feature_path)
        
        # Generate prediction
        predicted_caption = greedy_search_prediction(video_features, audio_features, tokenizer, encoder_model, decoder_model)
        
        # Store results
        ground_truth_captions = test_videos[video_id]
        results.append({
            'video_id': video_id,
            'predicted_caption': predicted_caption,
            'ground_truth_1': ground_truth_captions[0] if len(ground_truth_captions) > 0 else '',
            'ground_truth_2': ground_truth_captions[1] if len(ground_truth_captions) > 1 else '',
            # Add more ground truth columns if you wish
        })

    # Convert results to a pandas DataFrame and save as CSV
    results_df = pd.DataFrame(results)
    output_path = os.path.join(config.BASE_DIR, 'evaluation_results.csv')
    results_df.to_csv(output_path, index=False)
    
    print(f"\n--- Evaluation Complete ---")
    print(f"Results saved to: {output_path}")

if __name__ == '__main__':
    main()