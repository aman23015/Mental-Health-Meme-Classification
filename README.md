# Mental-Health-Meme-Classification
FUSE-MH: Figurative-understanding and Semantic Embedding for Mental Health memes

This project aims to classify mental health symptoms (specifically anxiety symptoms) from meme images using a Retrieval-Augmented Generation (RAG) framework enriched with OCR text, visual description, and figurative commonsense reasoning.

##  Repository Contents

- **Blip_VD.py**: Generates visual descriptions for memes using BLIP.
- **Knowledge_Fus.py**: Creates the Knowledge Fusion database combining OCR text, visual description, and figurative reasoning.
- **OCR.py**: Extracts text from meme images using EasyOCR.
- **llama_FR.py**: Generates figurative reasoning triples (Cause-Effect, Figurative Understanding, Mental State) using LLaMA-3.3-70B-Instruct (via NVIDIA NIM).
- **main.py**: Main training script for the BERT-based classifier using retrieval-augmented prompts.
- **inference.py**: Inference script to load the best model and evaluate it on test data.
- **inference_results.json**: Stores inference predictions and evaluation metrics.
- **Report.pdf**: Detailed technical report of the project (problem, methodology, results).
- **NLP_Project.pptx**: Final project presentation slides.
- **README.md**: This file.

