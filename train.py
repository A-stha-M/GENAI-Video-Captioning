# train.py (Final, Corrected Version)

import os
import json
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Concatenate, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import config

# --- GPU Configuration (Optional but Recommended) ---
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


def load_captions(train_val_path):
    """Loads captions and splits video IDs into train/validation sets."""
    with open(train_val_path, 'r') as f:
        train_val_data = json.load(f)
    
    captions = {}
    for sentence_item in train_val_data['sentences']:
        video_id = sentence_item['video_id']
        caption = f"<bos> {sentence_item['caption']} <eos>"
        if video_id not in captions:
            captions[video_id] = []
        captions[video_id].append(caption)

    train_ids = [v['video_id'] for v in train_val_data['videos'] if v['split'] == 'train']
    valid_ids = [v['video_id'] for v in train_val_data['videos'] if v['split'] == 'validate']
    
    return train_ids, valid_ids, captions


def create_tokenizer(captions, train_ids):
    """Creates and fits a tokenizer on the training captions."""
    train_captions = []
    for video_id in train_ids:
        train_captions.extend(captions.get(video_id, []))
        
    # --- FINAL FIX: Modify the filters to keep '<' and '>' for our special tokens ---
    filters = '!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n' # Default filters minus '<' and '>'
    tokenizer = Tokenizer(num_words=config.VOCAB_SIZE, oov_token="<unk>", filters=filters)
    tokenizer.fit_on_texts(train_captions)
    
    os.makedirs(config.SAVED_MODELS_DIR, exist_ok=True)
    with open(os.path.join(config.SAVED_MODELS_DIR, 'tokenizer.pkl'), 'wb') as f:
        pickle.dump(tokenizer, f)
        
    return tokenizer


def data_generator(video_ids, captions, tokenizer, batch_size):
    """A generator that yields batches of data for training."""
    vocab_size = tokenizer.num_words or config.VOCAB_SIZE
    
    while True:
        np.random.shuffle(video_ids)
        
        for i in range(0, len(video_ids), batch_size):
            batch_video_ids = video_ids[i:i + batch_size]
            
            batch_video_features = []
            batch_audio_features = []
            batch_decoder_input = []
            batch_decoder_output = []

            for video_id in batch_video_ids:
                video_feat_path = os.path.join(config.VIDEO_FEATURES_DIR, f"{video_id}.npy")
                audio_feat_path = os.path.join(config.AUDIO_FEATURES_DIR, f"{video_id}.npy")

                if not os.path.exists(video_feat_path) or not os.path.exists(audio_feat_path):
                    continue

                video_feats = np.load(video_feat_path)
                audio_feats = np.load(audio_feat_path)
                
                video_captions = captions.get(video_id, [])
                for caption in video_captions:
                    seq = tokenizer.texts_to_sequences([caption])[0]
                    
                    if len(seq) < 2:
                        continue

                    input_seq = seq[:-1]
                    target_seq = seq[1:]
                    
                    input_seq = pad_sequences([input_seq], maxlen=config.MAX_CAPTION_LENGTH, padding='post')[0]
                    target_seq = pad_sequences([target_seq], maxlen=config.MAX_CAPTION_LENGTH, padding='post')[0]

                    target_seq_one_hot = to_categorical(target_seq, num_classes=vocab_size)
                    
                    batch_video_features.append(video_feats)
                    batch_audio_features.append(audio_feats)
                    batch_decoder_input.append(input_seq)
                    batch_decoder_output.append(target_seq_one_hot)
            
            if len(batch_video_features) > 0:
                yield ((np.array(batch_video_features), np.array(batch_audio_features), np.array(batch_decoder_input)), np.array(batch_decoder_output))


def build_model(vocab_size):
    """Builds the audio-visual encoder-decoder model."""
    # --- Encoder ---
    video_input = Input(shape=(config.VIDEO_FRAMES, config.VIDEO_FEATURE_DIM), name='video_input')
    audio_input = Input(shape=(config.AUDIO_TIME_STEPS, config.AUDIO_FEATURE_DIM), name='audio_input')

    concatenated_features = Concatenate(axis=-1)([video_input, audio_input])
    
    encoder_lstm = LSTM(config.LATENT_DIM, return_state=True, name='encoder_lstm')
    _, state_h, state_c = encoder_lstm(concatenated_features)
    encoder_states = [state_h, state_c]

    # --- Decoder ---
    decoder_input = Input(shape=(None,), name='decoder_input')
    
    embedding_layer = Embedding(input_dim=vocab_size, output_dim=256, mask_zero=True)
    decoder_embedding = embedding_layer(decoder_input)
    
    decoder_lstm = LSTM(config.LATENT_DIM, return_sequences=True, return_state=True, name='decoder_lstm')
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
    
    dropout_layer = Dropout(0.5)
    decoder_outputs = dropout_layer(decoder_outputs)

    dense_layer = Dense(vocab_size, activation='softmax', name='decoder_output')
    decoder_outputs = dense_layer(decoder_outputs)

    # --- Full Trainable Model ---
    model = Model(inputs=[video_input, audio_input, decoder_input], outputs=decoder_outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # --- Inference Models (for prediction) ---
    encoder_model = Model(inputs=[video_input, audio_input], outputs=encoder_states)

    decoder_state_input_h = Input(shape=(config.LATENT_DIM,))
    decoder_state_input_c = Input(shape=(config.LATENT_DIM,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    
    inf_decoder_embedding = embedding_layer(decoder_input)

    inf_decoder_outputs, inf_state_h, inf_state_c = decoder_lstm(inf_decoder_embedding, initial_state=decoder_states_inputs)
    inf_decoder_states = [inf_state_h, inf_state_c]
    inf_decoder_outputs = dense_layer(inf_decoder_outputs)
    
    decoder_model = Model(inputs=[decoder_input] + decoder_states_inputs, outputs=[inf_decoder_outputs] + inf_decoder_states)
    
    return model, encoder_model, decoder_model


def main():
    """Main function to run the training process."""
    print("--- Starting Model Training ---")

    print("1. Loading data and creating tokenizer...")
    train_ids, valid_ids, captions = load_captions(config.TRAIN_VAL_ANNOTATION_FILE)
    tokenizer = create_tokenizer(captions, train_ids)
    vocab_size = tokenizer.num_words or config.VOCAB_SIZE
    if vocab_size < config.VOCAB_SIZE:
        vocab_size = config.VOCAB_SIZE
    print(f"Vocabulary size: {vocab_size}")

    print("2. Building model architecture...")
    model, encoder_model, decoder_model = build_model(vocab_size)
    model.summary()

    print("3. Setting up data generators...")
    train_generator = data_generator(train_ids, captions, tokenizer, config.BATCH_SIZE)
    valid_generator = data_generator(valid_ids, captions, tokenizer, config.BATCH_SIZE)
    
    steps_per_epoch = sum(len(captions.get(vid, [])) for vid in train_ids) // config.BATCH_SIZE
    validation_steps = sum(len(captions.get(vid, [])) for vid in valid_ids) // config.BATCH_SIZE
    if steps_per_epoch == 0: steps_per_epoch = 1
    if validation_steps == 0: validation_steps = 1
        
    print("4. Training model...")
    checkpoint_path = os.path.join(config.SAVED_MODELS_DIR, 'model_best.weights.h5')
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, save_weights_only=True, mode='min', verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

    model.fit(train_generator,
              epochs=config.EPOCHS,
              steps_per_epoch=steps_per_epoch,
              validation_data=valid_generator,
              validation_steps=validation_steps,
              callbacks=[checkpoint, early_stopping])
    
    print("\nLoading best weights for final model saving...")
    model.load_weights(checkpoint_path)

    print("5. Saving final inference models...")
    # --- FINAL FIX: Corrected typo from SAVED_MOCDELS_DIR to SAVED_MODELS_DIR ---
    encoder_model.save(os.path.join(config.SAVED_MODELS_DIR, 'encoder_model.h5'))
    decoder_model.save(os.path.join(config.SAVED_MODELS_DIR, 'decoder_model.h5'))
    print(f"Models saved in: {config.SAVED_MODELS_DIR}")
    
    print("\n--- Training Complete ---")

if __name__ == '__main__':
    main()