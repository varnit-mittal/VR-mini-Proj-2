# VR Mini Project 2

This repository implements a multiple‐choice Visual Question Answering (VQA) pipeline built on top of the Amazon Berkeley Objects (ABO) dataset. The core objectives are:

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
   - Created our own valuation metrics like CNS, VTGS



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

[Report](https://linktodocumentation)


## Environment Variables

To run `partA.ipynb` for dataset curation, you will need to add the following environment variables to your .env file

`KEYS` : key1,key2,key3,key4


## Run Locally

Clone the project

```bash
  git clone https://github.com/varnit-mittal/VR-mini-Proj-2
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
  python inference.py
```


## Authors

- [@varnit-mittal](https://www.github.com/varnit-mittal)
- [@ap5967ap](https://www.github.com/ap5967ap)
- [@meikenofdarth](https://www.github.com/meikenofdarth)

