# VR Mini Project 2 Overview

This project explores building a Visual Question Answering (VQA) system using the Amazon Berkeley Objects (ABO) dataset. We implemented zero-shot baselines with BLIP-2 and ViLT, then fine-tuned the BLIP-2 model using Low-Rank Adaptation (LoRA). Quantization techniques were also applied to reduce memory usage. We introduced two novel evaluation metrics—CSNS and VTGS—to better assess semantic and visual grounding in answers.


1. **Dataset Construction (Part A)**
   - Extract object-centric images and metadata from ABO 
   - Automatically generate multiple‐choice QA pairs targeting attributes such as color, shape, and material

2. **Baseline Evaluation (Part B)**
   - Train and evaluate standard VQA models (e.g., BLIP, ViLT) on the newly created dataset
   - Report accuracy and BERT score to understand Baseline models' performance

3. **LoRA‐based Fine‐Tuning (Part C)**
   - Apply Low‐Rank Adaptation (LoRA) to reduce the number of trainable parameters
   - Report accuracy and BERT score to understand Baseline models' performance

4. **Performance Analysis**
   - Use standard metrics (accuracy, precision, recall, F1)  
   - Created our own valuation metrics like CSNS, VTGS


# These commands will work for only WSL users. 
We are working on making them functional for Mac users too.
## API Reference

#### Get all items

```http
  Gemini API
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `api_key` | `string` | **Required**. Your API key |
| `MODEL ID` | `string` | **Required**. Model which you want to use. (Ours : gemini-2.5-flash-preview-04-17)|
| `prompt` | `string` | **Required**. Prompt to the model |
| `image` | `Pillow Image` | **Optional**. For Vision based tasks |




## Report

[Report](https://github.com/varnit-mittal/VR-mini-Proj-2/blob/main/Report.pdf)


## Environment Variables

To run `partA.ipynb` for dataset curation, you will need to add the following environment variables to your .env file

`KEYS` : key1,key2,key3,key4


## Run Locally

Clone the project

```bash
  git clone https://github.com/varnit-mittal/VR-mini-Proj-2.git
```

Go to the project directory

```bash
  cd VR-mini-Proj-2
```

Create and Activate a conda environment
```bash
  conda create -n myenv python=3.10
  conda activate myenv
```

Install dependencies

```bash
  pip install -r requirements.txt
```

Run the inference file

```bash
  python inference.py --image_dir /path/to/your/image/directory --csv_path /path/to/csv/file
```

## Authors

- [@varnit-mittal](https://www.github.com/varnit-mittal)
- [@ap5967ap](https://www.github.com/ap5967ap)
- [@meikenofdarth](https://www.github.com/meikenofdarth)


# About the Project
## Dataset

We use the small variant of the Amazon Berkeley Objects (ABO) dataset:

- ~147,000 product listings
- ~398,000 images
- Multilingual metadata (title, brand, description, etc.)

For curation, we used Gemini 2.5 Flash API with prompt templates to generate image-question-answer triples.


## Evaluation Metrics

Besides Accuracy and BERTScore, we used:

- **CSNS (Contextual Semantic & Numeric Score)**: Handles semantic similarity and numeric accuracy.
- **VTGS (Visual-Textual Grounded Score)**: Evaluates both textual alignment and visual grounding.

These allow more nuanced performance measurement beyond simple match scores.


## Sample Predictions

### ✅ Correct:
**Q:** What color is the basket?  
**Predicted:** Brown  
**Image:** `9b/9b147e56.jpg`

### ❌ Incorrect:
**Q:** What specific color term is used for the sparkles?  
**Predicted:** Silver  
**Expected:** Gold  
**Image:** `eb/ebc601ed.jpg`

## Limitations

- Fine-grained distinctions (e.g., between silicone and plastic) are difficult.
- One-word answers limit the complexity of questions.

## Future Work

- Multi-token answer generation
- Feedback-driven training with adaptive prompt difficulty
- Exploring more robust visual attention mechanisms


## References

- BLIP-2: https://huggingface.co/docs/transformers/model_doc/blip
- LoRA: https://arxiv.org/abs/2106.09685
- Gemini API: https://ai.google.dev/gemini-api/docs
- CSNS & VTGS: Custom-designed metrics for this project.
