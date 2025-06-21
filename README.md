# Audio-Visual Video Captioning with MSR-VTT

This project implements an encoder-decoder deep learning model to generate descriptive English captions for video clips. It leverages a novel fusion approach by using both **visual features** from video frames and **audio features** from the corresponding sound, allowing the model to generate richer, more context-aware descriptions. The model is trained and evaluated on the large-scale MSR-VTT dataset.

## Key Features

* **Dual-Modality Input:** Processes both video and audio streams to understand content more comprehensively.
* **VGG16 for Visual Features:** Utilizes a pre-trained VGG16 model to extract high-level features from sampled video frames.
* **MFCC for Audio Features:** Uses Librosa to extract Mel-Frequency Cepstral Coefficients (MFCCs), a robust representation of audio signals.
* **LSTM-based Encoder-Decoder:** Employs an LSTM-based architecture to encode the fused features and decode them into a sequence of words.
* **Modular & Configurable:** The entire project is structured with a central `config.py` file, making it easy to manage paths and hyperparameters.
* **Evaluation Pipeline:** Includes a script to run inference on the test set and save the predicted captions alongside ground-truth captions in a structured `.csv` file for easy analysis.

---

## Folder Structure

For the scripts to work correctly, your project must follow this directory structure:

---

## Setup and Installation

### 1. Prerequisites
* Python 3.8+
* Pip

### 2. Clone the Repository
Clone this project repository to your local machine.
```bash
git clone <your-repository-url>
cd GENAI-Video-Captioning

### 3. Install Dependencies
All required Python libraries should be in requirements.txt. Install them using pip:
```bash
pip install -r requirements.txt



