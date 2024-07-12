# Danbooru Tags to Natural Language Description using T5 Sequence-to-Sequence Model

This project utilizes Google's T5 model, finetuned to convert Danbooru tags into natural language descriptions.

## Requirements
python3, pytorch, huggingface's transformer library, bitsandbytes

## Getting Started

1. **Download the Model**: [Model Download Link](https://drive.google.com/file/d/1b96A3gxeoZZXish-Ct58qwRLplJhAOMd/view?usp=sharing)
   - Download the model and place it in the project directory alongside the scripts.

2. **Running Inference**
   - Execute `inference.py` with a string of Danbooru tags as an argument to generate a description.

   ```bash
   python3 inference.py "1girl, blue hair, hatsune miku, 3d, full body, short, girl, portrait, smiling"
   ```
  
   ```bash
   Input: 1girl, blue hair, hatsune miku, 3d, full body, short, girl, portrait, smiling
   Generated description: Full body 3d render of a short girl named hatsune miku with blue hair and smiling
   ```

3. **Training**
  - Modify the hyperparameters inside train.py as you see fit.
  - run train.py
